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

sys.path.append('evaluation/')
from intoxicat_evaluation import *


class Simple_Neural_Network(nn.Module):

    def __init__(self, input_size, layers_sizes, num_classes, dropout, bn, activation):
        super().__init__()
        self.input_size = input_size
        self.layers_sizes = layers_sizes
        self.num_classes = num_classes
        # try to make the number of linear layers dynamic
        self.layers = nn.ModuleList()
        input_size = self.input_size
        for size in self.layers_sizes:
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
        for size in self.layers_sizes:
            self.batch_norms.append(torch.nn.BatchNorm1d(size))
        self.batch_norms.append(torch.nn.BatchNorm1d(self.num_classes))

        # add dropout
        self.dropout = nn.Dropout(p=dropout)

        # add loss curve so loss can be stored in model directly
        self.loss_curve = []


    def forward(self, input_data, input_lengths, dropout, bn):

        # go through all of the linear layers
        input_linear_activation = torch.stack([inp[0] for inp in list(input_data)] )
        for i, linear_layer in enumerate(self.layers):
            input_linear = input_linear_activation
            # add dropout if in parameters
            if dropout:
                input_linear = self.dropout(input_linear)
            input_linear = linear_layer(input_linear)
            # add batch normalization if in parameters
            if bn:
                batch_norm = self.batch_norms[i]
                input_linear = batch_norm(input_linear)
            input_linear_activation = self.activation(input_linear)
            # print(f'Output after Linear Layer {i}: {input_linear_activation}')

        # put output from linear layer through linear activation activation function and put the result through our output layer
        # the output layer maps the representation of the linear layer to the classes that we want to choose from
        # output_fully_connected_layer_activation = self.activation(output_fully_connected_layer)
        output_fully_connected_layer_activation = input_linear_activation
        # print(f'Output after another activation: {output_fully_connected_layer_activation}')
        
        final_output_wo_softmax = self.output_layer(output_fully_connected_layer_activation)
        # print(f'Output after output layer: {final_output_wo_softmax}')

        if bn:
            last_batch_norm = self.batch_norms[-1]
            final_output_wo_softmax = last_batch_norm(final_output_wo_softmax)
        # add softmax function
        final_output = self.softmax(final_output_wo_softmax)
        # print(f'Final output: {final_output}')

        return final_output


    def store_loss(self, loss):
        self.loss_curve.append(loss) 