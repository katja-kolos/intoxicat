import os, json, sys
import parselmouth
import audiofile
import opensmile
import numpy as np

sys.path.append('..')
from basics import read_json


def extract_features(audiofile):

    features = {}
    snd = parselmouth.Sound(audiofile)
    pitch = snd.to_pitch().selected_array['frequency']
    intensity = snd.to_intensity()
    formants = snd.to_formant_burg(max_number_of_formants=3)

    mfcc = snd.to_mfcc(number_of_coefficients=12).to_array()

    for el in mfcc:
        print(el)

    f1, f2, f3 = [], [], []

    for t in formants.ts():
        if intensity.get_value(t) > 0:
            f1.append(formants.get_value_at_time(1, t))
            f2.append(formants.get_value_at_time(2, t))
            f3.append(formants.get_value_at_time(3, t))
        else:
            f1.append(np.nan)
            f2.append(np.nan)
            f3.append(np.nan)

    print(mfcc)
    features['pitch'] = pitch
    features['intensity'] = intensity.values

    # print(pitch)

    return features


def extract_features_opensmile(annotation_json, lld_json_name, functionals_json_name):

    json_lld = {}
    json_functionals = {}

    annotation_file = read_json(annotation_json)
    # acces every file
    for file, meta_data in annotation_file.items():

        print(file)
        audio_file = meta_data['annotates']
        path = meta_data['path']
        path_to_file = os.path.join(path, audio_file)
        print(f'Working on {audio_file}')
        # create key + nested dictionary for file
        json_lld[path_to_file] = {}
        json_functionals[path_to_file] = {}

        intoxication_status = meta_data['alc']
        json_lld[path_to_file]['intoxicated'] = intoxication_status
        json_functionals[path_to_file]['intoxicated'] = intoxication_status

        aak_status = meta_data['aak']
        json_lld[path_to_file]['breath alcohol concentration'] = aak_status
        json_functionals[path_to_file]['breath alcohol concentration'] = aak_status

        bak_status = meta_data['bak']
        json_lld[path_to_file]['blood alcohol concentration'] = bak_status
        json_functionals[path_to_file]['blood alcohol concentration'] = bak_status

        # create nested dictionary for the features
        json_lld[path_to_file]['features'] = {}
        json_functionals[path_to_file]['features'] = {}

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
            json_lld[path_to_file]['features'][name] = feature_list

        # build feature extractor
        functionals_feature_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        # extract features
        functionals_features = functionals_feature_extractor.process_signal(signal,sampling_rate)

        # extract feature names Functionals
        feature_names_functionals = functionals_feature_extractor.feature_names
        for name in feature_names_functionals:
            feature = functionals_features[name]
            feature_list = list(feature)
            json_functionals[path_to_file]['features'][name] = feature_list

    with open(lld_json_name, 'w') as jsn:
        json.dump(json_lld, jsn)

    with open(functionals_json_name, 'w') as jsn:
        json.dump(json_functionals, jsn)


if __name__ == "__main__":

    extract_features_opensmile('../data/meta_data_annotation_all_features_050623.json', 'ALC_features_LLD.json', 'ALC_features_Functional.json')
