import json, os, random, torch, sys, time, argparse
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

sys.path.append('preprocess/')
from prepare_data import split_dataset_into_splits

sys.path.append('evaluation/')
from intoxicat_evaluation import *

label_dict = {'a': 1, 'na': 0, 'cna': 0}


def get_keep_features(func_or_lld):

    with open('keep_features.tsv', 'r') as kf:
        lines = kf.readlines()

    if func_or_lld.lower() == 'functional':
        return [el.split('\t')[0].strip() for el in lines[1:] if not el.split('\t')[0].startswith('#')]
    elif func_or_lld.lower() == 'lld':
        return [el.split('\t')[1].strip() for el in lines[1:] if not el.split('\t')[1].startswith('#')]
    else:
        print('Invalid option: {}'.format(func_or_lld))
    

class Dataset:

    def __init__(self, path_to_file, func_or_lld):

        with open(path_to_file, 'r') as ptf:
            # load json file as dictionary
            json_file = json.load(ptf)

        # create empty lists to collect labels, features from all files and feature names
        labels, all_features, feature_names = [], [], []

        # iterate over the json dictionary
        # use enumerate to keep track of the index
        for i, (file_name,inner_dict) in enumerate(json_file.items()):
            # create empty list for the features of this file
            features = []
            # iterate over the inner dictionary
            # keys are 'intoxicated' and 'features'
            for key,value in inner_dict.items():
                if key == 'intoxicated':
                    # append label of this file (value) to labels
                    # transform string value into integer
                    intox_value = label_dict[value]
                    # add label
                    labels.append(intox_value)
                elif key == 'features':
                    # value of features is a dictionary 
                    # we want to collect the individual features in a list
                    # we have to make sure that the order of the features is always the same
                    # so we use sorted to sort the feature names and iterate over the dictionary using the sorted names
                    keep_features = get_keep_features(func_or_lld)
                    # print(keep_features)
                    # print(len(keep_features))
                    for feature_name in sorted(value.keys()):
                        # add feature to the feature list of this file if they are in the keep_features list
                        # print(feature_name)
                        if feature_name in keep_features:
                            # print('yes')
                            # access feature using the feature name from the sorted list
                            features.append(value[feature_name])

                        # we want to extract the feature names, but they are the same for all files
                        # so we only need to extract them once
                        # so we only extract them from the first file i.e. file of index 0 
                        if i == 0:
                            feature_names.append(feature_name)

                    # exit()
            # add the features from this file to the list that includes the features from all files
            # all_features has the following structure: [[[feature1], [feature2], …], [[feature1], [feature2], …], …]   
            # use torch.tensor() to transform feature list to a matrix
            all_features.append(torch.tensor(features))
            
        # equate our collected labels, features and feature names with the properties of the class
        # transform labels into one hot encodings (not necessary for binary classification,
        # but we keep it in case we want to switch to multiclass classification)  
        self.labels = F.one_hot(torch.tensor(labels))
        self.features = all_features
        self.feature_names = feature_names

    def __len__(self):
        # length of our dataset is the same as the length of our labels
        return len(self.labels)

    def __getitem__(self, idx):
        # get label of file at index idx
        label = self.labels[idx]
        # get feature matrix of file at index idx
        file_features = self.features[idx]

        # return both items as a tuple
        return (label, file_features, len(file_features[0]))


# train_corpus = Dataset('path/to/file')
# train_corpus.labels # will give use the labels of our dataset
# train_corpus.features # will give use the features of our dataset
# train_corpus.feature_names # will give use the names of the features in our dataset

def collate_costum(batch):
    # batch is a list of triples with labels, file features and length
    label_list, feature_list, feature_length_list = [], [], []
    
    # we want a batch of the three
    # iterate over batch
    # each element represents a file
    for element in batch:
        # add to lists for each element
        label_list.append(element[0])
        feature_list.append(element[1])
        feature_length_list.append(element[2])

    # stack the label one hot encodings
    stacked_labels = torch.stack(label_list)

    # get number of features 
    number_of_features = len(feature_list[0])
    # flatten current feature list so that it becomes a list of list 
    # instead of a list of lists of matrices
    # this is necessary in order to use the padding function
    flattened_feature_list = [feature_tensor for file_features in feature_list for feature_tensor in file_features]
    # use torch's padding function on flattened list
    flattened_stacked_features = pad_sequence(flattened_feature_list, batch_first=True)
    # transform the padded flattened tensor to a tensor of matrices
    # number of features indicate how many rows belong to one matrix (i.e. file)
    # permute shape of tensor to make it accessible for the model
    stacked_features = torch.permute(torch.stack(torch.split(flattened_stacked_features, number_of_features)), (0, 2, 1))
    
    # why do we need the length?
    stacked_feature_lengths = torch.LongTensor(feature_length_list)

    return stacked_labels, stacked_features, stacked_feature_lengths

    
# ---------------------------------------------------------------------

class LSTM_Model(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_size_2, num_layers, num_classes):
        # initialise init function of parent module
        super(LSTM_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.num_classes = num_classes

        # initialise LSTM architecture
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # map last hidden layer to intermediate layer before output layer
        self.fully_connected = torch.nn.Linear(self.hidden_size, self.hidden_size_2)
        # map intermediate layer to output layer
        self.output_layer = torch.nn.Linear(self.hidden_size_2, self.num_classes)

        # define a relu function
        self.relu = torch.nn.ReLU()

        # define a sigmoid function
        self.sigmoid = torch.nn.Sigmoid()

        # add some dropout
        self.dropout = torch.nn.Dropout(0.25)

    
    def forward(self, input_data, input_lengths):
        # use model and add any further layers and activation functions
        
        # pack the padded sequence so that the model knows where padding starts
        packed_input = pack_padded_sequence(input_data, input_lengths, batch_first=True, enforce_sorted=False)

        # put input data through our LSTM that we defined above
        packed_output, (final_hidden_state, final_cell_state) = self.lstm(packed_input)

        # unpack padded output
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # put output from LSTM through linear ReLU activation function and put the result trhough another linear layer
        # only take the output of the last time step (it's an RNN!) from each element in the batch
        output = torch.stack([el[-1] for el in output])
        output_relu = self.relu(output)

        # add some dropout for regularisation
        output_relu = self.dropout(output_relu)
        output_fully_connected_layer = self.fully_connected(output_relu)

        # put output from linear layer through linear ReLU activation function and put the result through our output layer
        # the output layer maps the representation of the linear layer to the classes that we want to choose from
        output_fully_connected_layer_relu = self.relu(output_fully_connected_layer)
        # add some dropout
        output_fully_connected_layer_relu = self.dropout(output_fully_connected_layer_relu)
        final_output_wo_sigmoid = self.output_layer(output_fully_connected_layer_relu)

        # add sigmoid function
        final_output = self.sigmoid(final_output_wo_sigmoid)

        return final_output


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Train an LSTM model')

    parser.add_argument('-s', '--split', action='store_true', help='Specify whether the data has already been split into train, validation and test set.')
    parser.add_argument('feature_files', type=str, nargs='+', help='If the data has not been split yet, specify the feature file and then the path where the split data should be put. If the data is already split, specify the path and the name of the file of the split files up until _(train|valid|test).json')
    parser.add_argument('model_file', type=str, help='Name of the model to be saved.')
    parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
    parser.add_argument('-t', '--test', action='store_true', help='Put this flag if you wish to test on the test set. Otherwise the model will be tested on the validation set.')
    args = vars(parser.parse_args())

    # python3 lstm_intoxicated_model.py (-s|--split) too_big_for_git/preprocess/ALC_features_Functional.json too_big_for_git/features
    # \ too_big_for_git/models/ALC_intoxicated_model_Functional.pt Functional

    print(parser.parse_args())

    if args['split']:

        paths = [args['feature_files'][0]]

        # split dataset into train, validation and test set
        split_dataset_into_splits(paths, [args['features']], args['feature_files'][-1])
        print('Splitted dataset')

        dir = args['feature_files'][-1]
        file_name = args['feature_files'][0].split('/')[-1].split('.')[0]

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

    print('Dataset loaded')

    # build model
    # input size = number of features (25)
    # hidden size = up to us (we are starting with 32)
    # number of layers = up to us (we are starting with 2)
    # number of classes = number of classes (we are starting with binary classification)
    model = LSTM_Model(len(train_dataset.feature_names), 32, 32, 2, 2)    
    model.to(device)
    print(model)

    # define loss function
    loss = torch.nn.BCELoss()

    # choose an optimizer
    # learning rate = up to us (we start with 0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # torch data loader for testing
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_costum, batch_size=2)

    # number of epochs = up to us (we are starting with 5)
    number_of_epochs = 5
    training_summary = []
    summary_freq_batches = 10
    
    # TRAINING LOOP:
    # ------------------------------------------------------

    print('Start training.')
    start = time.time()

    for epoch in range(number_of_epochs):
        print(f'EPOCH: {epoch}')
        for batch_no, (batch_labels, batch_file_features, batch_file_feature_lengths) in enumerate(train_loader):
            # no predictions yet because softmax is only applied by the loss function
            # therefore we get the logits from the model
            batch_file_features = batch_file_features.to(device)
            logits = model(batch_file_features, batch_file_feature_lengths)
            # calculate the current loss by comparing the predictions of the model with the actual labels
            sig_logs = logits.to(dtype=torch.float32, device=device)
            batch_labels = batch_labels.to(dtype=torch.float32, device=device)
            current_loss = loss(sig_logs, batch_labels)
            # set gradients to zero so that previous computations don't influence the computation of the current gardient(s)
            optimizer.zero_grad()
            # computes the gradients for backpropagation
            current_loss.backward()
            # use backpropagation to update your weights
            optimizer.step()
            if batch_no % summary_freq_batches == 0:
                training_summary.append(current_loss.item())
                print('------------------------------------')
                print(f'Batch no. {batch_no} => Loss: {current_loss}')
                print('------------------------------------')
    end = time.time()
    
    print('Stopped training! Training time was {}'.format(end - start))

    # save model
    torch.save(model.state_dict(), args['model_file'])

    # # load model
    # model = LSTM_Model(len(train_dataset.feature_names), 32, 32, 2, 2)  
    # model.load_state_dict(torch.load('../too_big_for_git/models/intoxication_model_{}.pt'.format(func_or_lld)))
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

    # print('Predictions: {}'.format(test_predictions))
    # print('\nLabels: {}'.format(test_labels))

    # transform labels and predictions for accuracy function
    test_labels_acc = [np.argmax(label.detach().numpy()) for label in test_labels]
    test_predictions_acc = [np.argmax(pred.cpu().detach().numpy()) for pred in test_predictions]

    # Compute the accuracy of the validation predictions
    print('\nTest accuracy after training: {:.2f}%'.format(calculate_accuracy(test_labels_acc, test_predictions_acc)*100))
        
    print('\nNumber of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))