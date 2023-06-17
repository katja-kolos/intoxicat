import argparse, sys, torch
import numpy as np
from lstm_intoxicated_model import Dataset, LSTM_Model, collate_costum
 
sys.path.append('evaluation/')
from intoxicat_evaluation import *

parser = argparse.ArgumentParser(description='Evaluate a trained model.')

parser.add_argument('data_path', type=str, help='Name of the test file.')
parser.add_argument('model_file', type=str, help='Name of the model to be loaded.')
parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
args = vars(parser.parse_args())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Loading datasets.')

test_dataset = Dataset(args['data_path'], args['features'])

print('Dataset loaded.')

# load model
model = LSTM_Model(len(test_dataset.feature_names), [32, 32], 2)
model.load_state_dict(torch.load(args['model_file']))
model.to(device)
model.eval()
print(model.eval())

# Test on test set
test_inputs = test_dataset.features

# load the test data set using a data loader
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_costum, batch_size=len(test_dataset))

# feed the inputs, fetch predictions
for batch_no, (batch_labels, batch_file_features, batch_file_feature_lengths) in enumerate(test_loader):
    batch_file_features = batch_file_features.to(device)
    test_predictions = model(batch_file_features, batch_file_feature_lengths).round()
    test_labels = batch_labels

pred_string = 'Prediction\t--\tLabel\n'
pred_string += '\n'.join(['{}\t--\t{}'.format(pred, test_labels[i]) for i, pred in enumerate(test_predictions)])
pred_file_name = '{}_predictions.txt'.format(args['model_file'].split('.pt')[0])

with open(pred_file_name, 'w') as pfn:
    pfn.write(pred_string)

# transform labels and predictions for accuracy function
test_labels_acc = [np.argmax(label.detach().numpy()) for label in test_labels]
test_predictions_acc = [np.argmax(pred.cpu().detach().numpy()) for pred in test_predictions]

# Compute the accuracy of the validation predictions
print('\nTest accuracy after training: {:.2f}%'.format(calculate_accuracy(test_labels_acc, test_predictions_acc)*100))

print('\nNumber of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))