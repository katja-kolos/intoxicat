import json, os, random, torch, sys, time, argparse, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import _LRScheduler

sys.path.append('preprocess/')
from prepare_data import split_dataset_into_splits


class LSTM_Model(nn.Module):

    def __init__(self, input_size, layers_sizes, num_layers, num_classes, dropout, bn, activation, bidirectional, bias):
        # initialise init function of parent module
        super(LSTM_Model, self).__init__()
        self.input_size = input_size
        # change hidden sizes for dynamic layers
        self.layers_sizes = layers_sizes
        self.num_layers = num_layers
        self.num_classes = num_classes

        # initialise LSTM architecture
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.layers_sizes[0], num_layers=self.num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout, bias=bias)
        
        # # map last hidden layer to intermediate layer before output layer
        # self.fully_connected = torch.nn.Linear(self.hidden_size, self.hidden_size_2)

        # try to make the number of linear layers dynamic
        self.layers = nn.ModuleList()
        input_size = self.layers_sizes[0]
        for size in self.layers_sizes[1:]:
            self.layers.append(torch.nn.Linear(input_size, size))
            input_size = size  # For the next layer

        # map intermediate layer to output layer
        self.output_layer = torch.nn.Linear(self.layers_sizes[-1], self.num_classes)

        # define an activation function
        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.nn.Sigmoid()

        # define a softmax function
        self.softmax = torch.nn.Softmax(dim=1)

        # define batch normalization
        # make it dynamic as well
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(torch.nn.BatchNorm1d(self.layers_sizes[0]))
        for size in self.layers_sizes[1:]:
            self.batch_norms.append(torch.nn.BatchNorm1d(size))
        self.batch_norms.append(torch.nn.BatchNorm1d(self.num_classes))

        # add dropout
        self.dropout = nn.Dropout(p=dropout) 

        # encode info about bidirectionality
        self.bidirectional = bidirectional

        # add loss curve so loss can be stored in model directly
        self.loss_curve = []


    def forward(self, input_data, input_lengths, dropout, bn):
        # use model and add any further layers and activation functions
        
        # pack the padded sequence so that the model knows where padding starts
        packed_input = pack_padded_sequence(input_data, input_lengths, batch_first=True, enforce_sorted=False)

        # if self.bidirectional:
        #     # initialise hidden & cell state
        #     h0 = torch.zeros(self.num_layers, input_data.size(0), self.layers_sizes[0]).to(input_data.device)
        #     c0 = torch.zeros(self.num_layers, input_data.size(0), self.layers_sizes[0]).to(input_data.device)
        # else:
        # initialise hidden & cell state
        h0 = torch.zeros(self.num_layers, input_data.size(0), self.layers_sizes[0]).to(input_data.device)
        c0 = torch.zeros(self.num_layers, input_data.size(0), self.layers_sizes[0]).to(input_data.device)
            
        # put input data through our LSTM that we defined above
        packed_output, _ = self.lstm(packed_input, (h0, c0))

        # unpack padded output
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # put output from LSTM through linear activation activation function and put the result trhough another linear layer
        # only take the output of the last time step (it's an RNN!) from each element in the batch

        output = torch.stack([el[input_lengths[i] - 1] for i, el in enumerate(output)])
        # print(f'Output after LSTM: {output}')

        if bn:
            # apply batch normalization
            first_batch_norm = self.batch_norms[0]
            output = first_batch_norm(output)

        output_activation = self.activation(output)
        # print(f'Output after activation: {output_activation}')

        # go through all of the linear layers
        input_linear_activation = output_activation
        for i, linear_layer in enumerate(self.layers):
            input_linear = input_linear_activation
            # add dropout if in parameters
            if dropout:
                input_linear = self.dropout(input_linear)
            input_linear = linear_layer(input_linear)
            # add batch normalization if in parameters
            if bn:
                batch_norm = self.batch_norms[i+1]
                input_linear = batch_norm(input_linear)
            input_linear_activation = self.activation(input_linear)
            # print(f'Output after Linear Layer {i}: {input_linear_activation}')

        # put output from linear layer through linear activation activation function and put the result through our output layer
        # the output layer maps the representation of the linear layer to the classes that we want to choose from
        output_fully_connected_layer_activation = input_linear_activation
        # print(f'Output after another activation: {output_fully_connected_layer_activation}')
        
        final_output_wo_softmax = self.output_layer(output_fully_connected_layer_activation)
        # print(f'Output after output layer: {final_output_wo_softmax}')

        if bn:
            # apply batch normalization
            last_batch_norm = self.batch_norms[-1]
            final_output_wo_softmax = last_batch_norm(final_output_wo_softmax)
        
        # add softmax function
        final_output = self.softmax(final_output_wo_softmax)
        # print(f'Final output: {final_output}')

        return final_output


        def store_loss(self, loss):
            self.loss_curve.append(loss) 