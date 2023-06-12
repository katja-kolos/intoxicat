import pandas as pd
import sys, json, os, random
from sklearn.model_selection import train_test_split


def create_file_wrapper(path, features, what_type, out_name, meta_data=None):

    file_annotations = {}

    # access every directory
    for directory in os.listdir(path):
        try:
            # access every file
            for file in os.listdir(path + directory):
                file_path = os.path.join(path, directory, file)
                if check_for_valid_file(os.listdir(path + directory), file, file_path):
                    print(f'Found valid JSON file: {file}')
                    
                    with open(file_path, 'r') as jsn:
                        annotation_dict = json.load(jsn)

                    file_annotations[file_path] = {}
                    file_annotations[file_path]['path'] = path + directory

                    for feature in features.split(','):
                        file_annotations[file_path][feature] = annotation_dict[feature]

                    if what_type == 'meta_data':
                        file_annotations = gather_metadata(meta_data, annotation_dict, file_annotations, file_path)
                    elif what_type == 'word_transcr':
                        file_annotations = gather_word_transcriptions(annotation_dict, file_annotations, file_path)
                    elif what_type == 'phonetic_transcr':
                        file_annotations = gather_phonetic_transcription(annotation_dict, file_annotations, file_path)
                    else:
                        print('INVALID ANNOTATION TYPE!')
                        exit()
        except OSError:
            continue

    print(f'Writing annotation file: {out_name}')
    with open(out_name, 'w') as jsn:
        json.dump(file_annotations, jsn)


def check_for_valid_file(files, file, path):
    # check if recorder file is a valid file
    # from the README:
    # - use only recordings where a BPF *.par or a *.TextGrid file exists, from these discard all multiple versions that carry the entries
    # ACO: false
    # ACO: false 2nd
    # ACO: spont
    # in the BPF file header (*.par).

    file_name, file_extension = os.path.splitext(file)

    valid = True
    if file_extension == '.json':
        if '_'.join(file_name.split('_')[:-1]) + '.TextGrid' in files:
            if '_'.join(file_name.split('_')[:-1]) + '.par' in files:
                with open('_'.join(path.split('_')[:-1]) + '.par', 'r') as par:
                    par_lines = par.readlines()
                par_lines_stripped = [line.strip() for line in par_lines]
                invalid_items = ['ACO: false', 'ACO: false 2nd', 'ACO: spont']
                for inv in invalid_items:
                    if inv in par_lines_stripped:
                        valid = False
            else:
                valid = False
        else:
            valid=False
    else:
        valid = False

    return valid


def gather_metadata(meta_data, annotation_dict, file_annotations, file_path):

    if meta_data != 'n':
        items = annotation_dict['levels'][0]['items']

        for item_dict in items:
            for name_val_pair in item_dict['labels']:
                if name_val_pair['name'] in meta_data.split(','):
                    file_annotations[file_path][name_val_pair['name']] = name_val_pair['value']

    return file_annotations           


def gather_word_transcriptions(annotation_dict, file_annotations, file_path):
    
    items = annotation_dict['levels'][1]['items']

    word_list, transcription_list = [], []

    for item_dict in items:
        for name_val_pair in item_dict['labels']:
            if name_val_pair['name'] == 'word':
                word_list.append(name_val_pair['value'])
            elif name_val_pair['name'] == 'cano':
                transcription_list.append(name_val_pair['value'])

    file_annotations[file_path]['words'] = word_list
    file_annotations[file_path]['transcription'] = transcription_list

    return file_annotations


def gather_phonetic_transcription(annotation_dict, file_annotations, file_path):

    items = annotation_dict['levels'][2]['items']

    word_list, transcription_list = [], []

    for item_dict in items:
        file_annotations[file_path][item_dict['id']] = {}
        file_annotations[file_path][item_dict['id']]['sample_start'] = item_dict['sampleStart']
        file_annotations[file_path][item_dict['id']]['sample_duration'] = item_dict['sampleDur']
        for name_val_pair in item_dict['labels']:
            if name_val_pair['name'] == 'phonetic':
                file_annotations[file_path][item_dict['id']]['sample_value'] = name_val_pair['value']

    return file_annotations


def create_toy_dataset(annotation_file, feature_files):

    df = pd.read_json(annotation_file, orient='index')
    train, test = train_test_split(df, test_size=0.01)

    print(len(test))
    print(test.columns)
    # for test_el in test:
    #     print(test_el)
    audio_files = [os.path.join(row['path'], row['annotates']) for index, row in test.iterrows()]

    for feature_file in feature_files:
        with open(feature_file, 'r') as ff:
            features = json.load(ff)
        feature_dict = {}
        print('Number of Audio Files: {}'.format(len(audio_files)))
        for i, audio_file in enumerate(audio_files):
            feature_dict[audio_file] = features[audio_file]

            if i % 100 == 0:
                print(i)

        new_name = feature_file[:-5] + '_toy.json'
        with open(new_name, 'w') as ff:
            json.dump(feature_dict, ff)


def split_dataset_into_splits(path_to_dataset, func_or_lld, out_path):
    with open(path_to_dataset[0], 'r') as f:
        dataset_1 = json.load(f)

    with open(path_to_dataset[1], 'r') as f:
        dataset_2 = json.load(f)

    # shuffle the keys of the dictionary
    keys = list(dataset_1.keys())
    random.shuffle(keys)

    # calculate the number of samples for each split
    num_samples = len(dataset_1)
    num_train = int(num_samples * 0.8)
    num_val = int(num_samples * 0.1)

    # create three new dictionaries for training, validation, and testing
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    # iterate over the shuffled keys and add each key-value pair to the appropriate dictionary
    for i, dataset in enumerate([dataset_1, dataset_2]):
        for j, key in enumerate(keys):
            if j < num_train:
                train_dict[key] = dataset[key]
            elif j < num_train + num_val:
                val_dict[key] = dataset[key]
            else:
                test_dict[key] = dataset[key]
        
        os.makedirs(out_name, exist_ok=True)
        # save the resulting dictionaries

        file_name = dataset.split('/')[-1].split('.')[0]

        with open("{}/{}_train.json".format(out_path, file_name), mode="w", encoding="utf8") as f:
            json.dump(train_dict, f)
        with open("{}/{}_val.json".format(out_path, file_name), mode="w", encoding="utf8") as f:
            json.dump(val_dict, f)
        with open("{}/{}_test.json".format(out_path, file_name), mode="w", encoding="utf8") as f:
            json.dump(test_dict, f)



if __name__ == "__main__":

    # command:
    # python3 prepare_data.py <path_to_directory> <basic_features> <annotation_type> <output_name> (meta_data=<meta_name>)

    # <path_to_directory>:  string that specifies the path to the directory containing the session directories

    # <basic_features>:     list of feature names, seperated by commas
    # possible features:    'name,annotates,sampleRate'

    # <annotation_type>:    string that states what type of annotation should be extracted
    # possible types:       'meta_data'|'word_transcr'|'phonetic_transcr'

    # <output_name>:        string that specifies the name of the output file

    # <meta_name>:          list of meta feature names, seperated by commas
    # possible features:    'utterance,utt,spn,o_utt,item,o_item,alc,sex,age,acc,drh,aak,bak,ges,ces,wea,irreg,anncom,specom,type,content'       

    try:
        create_file_wrapper(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    except IndexError:
        create_file_wrapper(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    # create_toy_dataset('../data/meta_data_annotation_all_features_220523.json', ['../../preprocess/ALC_features_Functional.json', '../../preprocess/ALC_features_LLD.json'])
