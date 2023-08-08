import json, os, random, torch, sys, time, argparse, copy, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import _LRScheduler

from trainloop_utilities import *
from basics import *

from preprocess.prepare_data import split_dataset_into_splits
from preprocess.data_utilities import Dataset, collate_costum

from evaluation.intoxicat_evaluation import *

from models.lstm_intoxicated_model import LSTM_Model
from models.simple_nn_intoxicated_model import Simple_Neural_Network


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Train an LSTM model')
parser.add_argument('-s', '--split', action='store_true', help='Specify whether the data has already been split into train, validation and test set.')
parser.add_argument('feature_files', type=str, nargs='+', help='If the data has not been split yet, specify the feature file and then the path where the split data should be put. If the data is already split, specify the path and the name of the file of the split files up until (_train|_valid|_test).json')
parser.add_argument('model_file', type=str, help='Name of the model to be saved.')
parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
parser.add_argument('parameters', type=str, default='Functional', help='Specify the parameters of the model.')
parser.add_argument('-t', '--test', action='store_true', help='Put this flag if you wish to test on the test set. Otherwise the model will be tested on the validation set.')
args = vars(parser.parse_args())

print('--------------------------------------------------------------------')
print(parser.parse_args())

if args['split']:
    paths = [args['feature_files'][0]]
    # split dataset into train, validation and test set
    split_dataset_into_splits(paths, [args['features']], args['feature_files'][-1])
    print('Splitted dataset')
    dir = args['feature_files'][-1]
    file_name = args['feature_files'][0].split('/')[-1].split('.')[0]
    print(file_name)

else:
    dir = '/'.join(args['feature_files'][-1].split('/')[:-1])
    file_name = args['feature_files'][0].split('/')[-1]

# load the datasets
print('Loading datasets.')
dir = dir[:-1] if dir.endswith('/') else dir
train_dataset = Dataset('{}/{}_train.json'.format(dir, file_name, args['features']), args['features'])
if args['test']:
    test_dataset = Dataset('{}/{}_test.json'.format(dir, file_name, args['features']), args['features'])
else:
    test_dataset = Dataset('{}/{}_valid.json'.format(dir, file_name, args['features']), args['features'])
print('Datasets loaded.')

# build model
params = json.loads(args['parameters'])
learning_rate = params['lr']
layers = params['layers']
dropout =   params['dropout']
optimizer = params['optim']
batch_norm = params['bn']
batch_size = params['batch_size']
activation = params['activation']
try:
    bidirectional = params['bidirectional']
    lstm_layers = params['lstm_layers']
    bias = params['bias']
except KeyError:
    pass

if args['features'] == 'Functional':
    # input size = number of features (88)
    # hidden layers = up to us (we are starting with [64, 32, 16, 8, 4])
    # number of classes = number of classes (we are starting with binary classification)
    # dropout
    # batch normalization
    model = Simple_Neural_Network(len(train_dataset.feature_names), layers, 2, dropout, eval(batch_norm), activation)
    out_file = 'parameters_and_results/snn_results.csv'
elif args['features'] == 'LLD':
    # input size = number of features (25)
    # hidden layers = up to us (we are starting with [16, 8, 4])
    # number of layers (LSTM) = up to us (we are starting with 2)
    # number of classes = number of classes (we are starting with binary classification)
    # dropout
    # bidirectional
    # batch normalization
    model = LSTM_Model(len(train_dataset.feature_names), layers, lstm_layers, 2, dropout, eval(batch_norm), activation, eval(bidirectional), eval(bias))    
    out_file = 'parameters_and_results/lstm_results.csv'
else:
    print('Invalid feature choice!')
    exit()

model.to(device)
print(model)

# define loss function
loss = torch.nn.BCELoss()
# choose an optimizerm (Adam (multiple options for later))
# learning rate = up to us (we start with 0.0001 and using Toucan warm-up scheduler)
if optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# balance batches
# torch data loader for testing
train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_costum, batch_size=batch_size, drop_last=True)
# warm up scheduler for learning rate
scheduler = ToucanWarmupScheduler(optimizer, warmup_steps=4000)
# number of epochs = up to us (we are starting with 10)
number_of_epochs = params['num_epochs']
training_summary = []
summary_freq_batches = 20

# TRAINING LOOP:
# ------------------------------------------------------
print('Start training.')
start = time.time()
weight_list_after = [copy.deepcopy(model.layers[i].weight) for i, layer in enumerate(model.layers)]
for epoch in range(number_of_epochs):
    print(f'EPOCH: {epoch}')
    for batch_no, (batch_labels, batch_file_features, batch_file_feature_lengths, batch_file_names) in enumerate(train_loader):
        # no predictions yet because softmax is only applied by the loss function
        # therefore we get the logits from the model
        batch_file_features = batch_file_features.to(device).requires_grad_(True)
        logits = model(batch_file_features, batch_file_feature_lengths, dropout, batch_norm)
        # calculate the current loss by comparing the predictions of the model with the actual labels
        sig_logs = logits.to(dtype=torch.float32, device=device)
        batch_labels = batch_labels.to(dtype=torch.float32, device=device)
        current_loss = loss(sig_logs, batch_labels)
        # computes the gradients for backpropagation
        current_loss.backward()
        # use backpropagation to update your weights
        optimizer.step()
        # set gradients to zero so that previous computations don't influence the computation of the current gardient(s)
        optimizer.zero_grad()
        # call warm-up scheduler
        scheduler.step()
        if batch_no % summary_freq_batches == 0:
            # store loss
            model.store_loss(current_loss.item())
            training_summary.append(current_loss.item())
            print('------------------------------------')
            print(f'Batch no. {batch_no} => Loss: {current_loss}')
            print('------------------------------------')
        
end = time.time()

print('Stopped training! Training time was {}'.format(end - start))
# save model
torch.save(model.state_dict(), args['model_file'])
# # load model
# model = LSTM_Model(len(train_dataset.feature_names), [32, 32], 2)  
# model.load_state_dict(torch.load('../too_big_for_git/models/intoxication_model_{}.pt'.format(func_or_lld)))
model.eval()
print(model.eval())
print('Now we start testing!')
# Test on test set
test_inputs = test_dataset.features
    
# load the test data set using a data loader
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_costum, batch_size=len(test_dataset))
# feed the inputs, fetch predictions
for batch_no, (batch_labels, batch_file_features, batch_file_feature_lengths, batch_file_names) in enumerate(test_loader):
    batch_file_features = batch_file_features.to(device)
    batch_file_names = batch_file_names
    test_predictions = model(batch_file_features, batch_file_feature_lengths, dropout, batch_norm).round()
    test_labels = batch_labels

pred_dict = {file_name: (int(torch.argmax(test_labels, dim=1)[i]), int(torch.argmax(test_predictions, dim=1)[i])) for i, file_name in enumerate(batch_file_names)}
pred_file_name = '{}/preds/{}_predictions.json'.format('/'.join(args['model_file'].split('/')[:-1]), args['model_file'].split('/')[-1].strip('.pt'))
write_json(pred_file_name, pred_dict)

# generate confusion matrix
plot_confusion_matrix(test_labels, test_predictions, args['model_file'])

# transform labels and predictions for accuracy function
test_labels_acc = [np.argmax(label.detach().numpy()) for label in test_labels]
test_predictions_acc = [np.argmax(pred.cpu().detach().numpy()) for pred in test_predictions]
# Compute the accuracy of the validation predictions
print('\nTest accuracy after training: {:.2f}%'.format(calculate_accuracy(test_labels_acc, test_predictions_acc)*100))

with open(out_file, 'a') as of:
    of.write('\n{}\t{}'.format(re.search('combo\d+', args['model_file'])[0], round(calculate_accuracy(test_labels_acc, test_predictions_acc)*100, 3)))
    
print('\nNumber of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print('--------------------------------------------------------------------')
