#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Laura & Francesca
# =============================================================================
"""This file contains code that is used to build a Dataset class, as well as the collate function to costumise the DataLoader"""
# =============================================================================
# Imports
# =============================================================================
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


label_dict = {'na': 0, 'cna': 0, 'a': 1}


def get_keep_features(func_or_lld):

    # this file includes the names of all of the extracted features
    # names that are marked by a "#" in front of them will be ignored 
    with open('keep_features.tsv', 'r') as kf:
        lines = kf.readlines()

    # return a list including all of the feature names that should be considered by the model 
    if func_or_lld.lower() == 'functional':
        return [el.split('\t')[0].strip() for el in lines[1:] if not el.split('\t')[0].startswith('#')]
    elif func_or_lld.lower() == 'lld':
        return [el.split('\t')[1].strip() for el in lines[1:] if not el.split('\t')[1].startswith('#')]
    else:
        print('Invalid option: {}'.format(func_or_lld))
    

class Dataset:
# define the dataset class that will be used by the models to access data

    def __init__(self, path_to_file, func_or_lld):

        with open(path_to_file, 'r') as ptf:
            # load json file as dictionary
            json_file = json.load(ptf)

        # create empty lists to collect labels, features from all files and feature names
        labels, all_features, feature_names, file_names = [], [], [], []

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
                    # get the list of the features that should be considered
                    keep_features = get_keep_features(func_or_lld)
                    # value of features is a dictionary 
                    # we want to collect the individual features in a list
                    # we have to make sure that the order of the features is always the same
                    # so we use sorted to sort the feature names and iterate over the dictionary using the sorted names
                    for feature_name in sorted(value.keys()):
                        # add feature to the feature list of this file if they are in the keep_features list
                        if feature_name in keep_features:
                            # access feature using the feature name from the sorted list
                            features.append(value[feature_name])

                        # we want to extract the feature names, but they are the same for all files
                        # so we only need to extract them once
                        # so we only extract them from the first file i.e. file of index 0 
                        if i == 0:
                            feature_names.append(feature_name)

                elif key == 'annotates':
                    # append filename of this file (value) to filenames
                    file_names.append('{}/{}'.format(inner_dict['path'], value))

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
        self.file_names = file_names

    def __len__(self):
        # length of our dataset is the same as the length of our labels
        return len(self.labels)

    def __getitem__(self, idx):
        # get label of file at index idx
        label = self.labels[idx]
        # get feature matrix of file at index idx
        file_features = self.features[idx]
        # get file name of file at index idx
        file_name = self.file_names[idx]

        # return both items as a tuple
        return (label, file_features, len(file_features[0]), file_name)


# train_corpus = Dataset('path/to/file')
# train_corpus.labels # will give use the labels of our dataset
# train_corpus.features # will give use the features of our dataset
# train_corpus.feature_names # will give use the names of the features in our dataset


def collate_costum(batch):
    # batch is a list of triples with labels, file features and length
    label_list, feature_list, feature_length_list, file_name_list = [], [], [], []
    
    # we want a batch of the three
    # iterate over batch
    # each element represents a file
    for element in batch:
        # add to lists for each element
        label_list.append(element[0])
        feature_list.append(element[1])
        feature_length_list.append(element[2])
        file_name_list.append(element[3])

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
    
    stacked_feature_lengths = torch.LongTensor(feature_length_list)

    return stacked_labels, stacked_features, stacked_feature_lengths, file_name_list