#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Laura
# =============================================================================
"""This file contains code to get interesting statistics from the meta data annotation file that is associated with the ALC data"""
# =============================================================================
# Imports
# =============================================================================
import json
import matplotlib.pyplot as plt
import numpy as np


def get_stats(feature, annotation_file):

    with open(annotation_file, 'r') as af:
        annotation_dict = json.load(af)

    feature_value_dict = {}
    for file, features in annotation_dict.items():
        if features[feature] in feature_value_dict:
            feature_value_dict[features[feature]] += 1
        else:
            feature_value_dict[features[feature]] = 1

    return feature_value_dict


def plot_stats_bar_chart(stat_dict): 

    # TODO: make this prettier!
    #       like sort xlabels alphabetically etc.

    value_list = [stat_dict[key] for key in list(stat_dict.keys())]
    key_list = list(stat_dict.keys())

    y_pos = np.arange(len(key_list))
    plt.bar(y_pos, value_list, align='center', alpha=0.5)
    plt.xticks(y_pos, key_list)
    # plt.ylabel('Usage')
    plt.show()



if __name__ == "__main__":

    # TODO: add argv options

    anno_file = '../data/meta_data_annotation_all_features_130623.json'

    mf_dict = get_stats('alc', anno_file)

    print(mf_dict)

    plot_stats_bar_chart(mf_dict)