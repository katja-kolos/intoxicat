import argparse, sys, torch, json, re
import numpy as np

from trainloop_utilities import *
from basics import *

from preprocess.data_utilities import Dataset, collate_costum

from evaluation.intoxicat_evaluation import *

from models.lstm_intoxicated_model import LSTM_Model
from models.simple_nn_intoxicated_model import Simple_Neural_Network


parser = argparse.ArgumentParser(description='Evaluate a trained model.')

parser.add_argument('data_path', type=str, help='Name of the test file.')
parser.add_argument('model_file', type=str, help='Name of the model to be loaded.')
parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
parser.add_argument('parameters', type=str, default='Functional', help='Specify the parameters of the model.')
args = vars(parser.parse_args())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Loading datasets.')

test_dataset = Dataset(args['data_path'], args['features'])

print('Dataset loaded.')

params = json.loads(args['parameters'])
layers = params['layers']
dropout =   params['dropout']
batch_norm = params['bn']
batch_size = params['batch_size']
activation = params['activation']
bidirectional = params['bidirectional']
lstm_layers = params['lstm_layers']
bias = params['bias']

# load model
model = LSTM_Model(len(test_dataset.feature_names), layers, lstm_layers, 2, dropout, eval(batch_norm), activation, eval(bidirectional), eval(bias))
model.load_state_dict(torch.load(args['model_file']))
model.to(device)
model.eval()
print(model.eval())

out_file = 'parameters_and_results/lstm_results.csv'

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
