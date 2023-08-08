#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Laura
# =============================================================================
"""This file contains the code to extract features from the ALC data"""
# =============================================================================
# Imports
# =============================================================================
import os, json, sys
import parselmouth
import audiofile
import opensmile
import numpy as np

sys.path.append('..')
from basics import read_json


def extract_features_opensmile(annotation_json, lld_json_name, functionals_json_name):
# extract both Functional and LLD features from audio files using opensmile

    json_lld = {}
    json_functionals = {}

    annotation_file = read_json(annotation_json)
    # access every file
    for file, meta_data in annotation_file.items():

        print(file)
        audio_file = meta_data['annotates']
        audio_file_tag = meta_data['name']
        path = meta_data['path']
        speaker_id = meta_data['spn']

        path_to_file_tag = os.path.join(path.split('/')[-1], audio_file_tag)
        path_to_file = os.path.join(path, audio_file)
        print(f'Working on {audio_file}')
        # create key + nested dictionary for file
        json_lld[path_to_file_tag] = {}
        json_functionals[path_to_file_tag] = {}

        # json_lld[path_to_file]['spn'] = speaker_id
        # json_functionals[path_to_file]['spn'] = speaker_id

        intoxication_status = meta_data['alc']
        json_lld[path_to_file_tag]['intoxicated'] = intoxication_status
        json_functionals[path_to_file_tag]['intoxicated'] = intoxication_status

        aak_status = meta_data['aak']
        json_lld[path_to_file_tag]['breath alcohol concentration'] = aak_status
        json_functionals[path_to_file_tag]['breath alcohol concentration'] = aak_status

        bak_status = meta_data['bak']
        json_lld[path_to_file_tag]['blood alcohol concentration'] = bak_status
        json_functionals[path_to_file_tag]['blood alcohol concentration'] = bak_status

        for key, value in meta_data.items():
            json_lld[path_to_file_tag][key] = value
            json_functionals[path_to_file_tag][key] = value

        # create nested dictionary for the features
        json_lld[path_to_file_tag]['features'] = {}
        json_functionals[path_to_file_tag]['features'] = {}

        # read audiofile into memory
        signal, sampling_rate = audiofile.read(path_to_file, always_2d=True)

        # build feature extractor
        lld_feature_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        # extract features
        lld_features = lld_feature_extractor.process_signal(signal,sampling_rate)

        # extract feature names LLDs
        feature_names_lld = lld_feature_extractor.feature_names
        for name in feature_names_lld:
            feature = lld_features[name]
            feature_list = list(feature)
            json_lld[path_to_file_tag]['features'][name] = feature_list

        # build feature extractor
        functionals_feature_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        # extract features
        functionals_features = functionals_feature_extractor.process_signal(signal,sampling_rate)

        # extract feature names Functionals
        feature_names_functionals = functionals_feature_extractor.feature_names
        for name in feature_names_functionals:
            feature = functionals_features[name]
            feature_list = list(feature)
            json_functionals[path_to_file_tag]['features'][name] = feature_list

    with open(lld_json_name, 'w') as jsn:
        json.dump(json_lld, jsn)

    with open(functionals_json_name, 'w') as jsn:
        json.dump(json_functionals, jsn)


if __name__ == "__main__":

    extract_features_opensmile('../data/meta_data_annotation_all_features_130623.json', '../../too_big_for_git/ALC_features_LLD.json', '../../too_big_for_git/ALC_features_Functional.json')
