import pandas as pd
import sys, json, os, random, argparse
from scipy.stats import zscore

# !!!!!!!!!!!
# works only for Functional features so far
# !!!!!!!!!!!

def get_speaker_dict(file_dict):

    speaker_dict = {}
    for audio_file, audio_annos in file_dict.items():
        try:
            speaker_dict[audio_annos['spn']].append(audio_file)
        except KeyError:
            speaker_dict[audio_annos['spn']] = [audio_file]

    return speaker_dict


def global_z_normalization(file_name, out_file, lld):

    with open(file_name, 'r') as fn:
        file_dict = json.load(fn)

    speaker_dict = get_speaker_dict(file_dict)

    for speaker, audio_files in speaker_dict.items():
        data_frames = []
        for audio_file in audio_files:
            data_frames.append(pd.DataFrame({**file_dict[audio_file]['features'], 'file_name': audio_file}))
            # print(pd.DataFrame({**file_dict[audio_file]['features'], 'file_name': audio_file}))        
            # exit()

        concatenated_df = pd.concat(data_frames)
        concatenated_df = concatenated_df.set_index('file_name')

        if len(data_frames) > 1:
            normalized_df = concatenated_df.apply(zscore)
        else:
            normalized_df = concatenated_df

        for audio_file in audio_files:
            if lld:
                audio_dict = {}
                for ind_audio_file_dict in normalized_df.loc[[audio_file]].to_dict('records'):
                    for key, value in ind_audio_file_dict.items():
                        if key in audio_dict:
                            audio_dict[key].append(value)
                        else:
                            audio_dict[key] = [value]
                file_dict[audio_file]['features'] = audio_dict
            else:   
                file_dict[audio_file]['features'] = {key: [value] for key, value in normalized_df.loc[[audio_file]].to_dict('records')[0].items()}

    with open(out_file, 'w') as of:
        json.dump(file_dict, of)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Normalize the features based on the speaker.')
    parser.add_argument('feature_file', type=str, help='Path to the unnormalized feature file.')
    parser.add_argument('normalized_feature_file', type=str, help='Name of the normalized feature file.')
    parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
    args = vars(parser.parse_args())

    global_z_normalization(args['feature_file'], args['normalized_feature_file'], args['features'])
