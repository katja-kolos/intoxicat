#!/usr/bin/env python
# coding: utf-8

'''
Hey :) This script looks at metadata, Functional features OR LLD features, 
and returns datasamples that satisfy your needs.

The output by default is the contents of the corresponding features file, 
but you can also keep fields from the metadata json (preserve_meta_data=True).

You define what part of the dataset you're interested in by: 
[from the command line] a sequence of 1 or more filter strings;
[when importing the create_subset function] a list of triples.

Each filter is a tuple: argument, operator, value. Supported operators:
> (alternative: gt), < (alternative: lt), == (alternative: eq), isin.

Arguments within the triple string are separated by ,. 

Attention! For two or more different tuples of filters, 
the condition is interpreted as 'and' (intersection of the subconditions).
'''


def create_subset(filters, 
                  save_path='preprocess/', 
                  save_df=True, 
                  return_df=False, 
                  preserve_meta_data=False,
                  features='Functional',
                  balance_classes=True, 
                  max_samples=None
                 ):
    import json
    import pandas as pd
    import numpy as np
    import os
    import stat
    #receives filters as [('age', '>', 30), ('sex', '==', 'F')]
    #creates a json file in intoxicat/preprocess
    
    print(filters)
    flag_preserve_metadata = 'meta_' if preserve_meta_data else ''
    flag_balanced = 'balanced_' if balance_classes else ''
    flag_max_samples = f'{max_samples}_samples' if max_samples else ''
    flags = ''.join([flag_preserve_metadata, flag_balanced, flag_max_samples])
    print('Additional requirements: ', flag_preserve_metadata, flag_balanced, flag_max_samples)
    
    ### helper functions
    def preprocess_triple(triple):
        
        def adjust_value_type(value):
            value = value.strip().strip('"').strip("'")
            #check the type and convert to the one in the dataframe
            value_exp_type = type(meta_data_df[arg][0])
            return value_exp_type(value) 
        
        arg, operator, value = triple
        if arg not in meta_data_df.columns:
            print (f'ERROR: no such column in the metadata: {arg}')
            print (f'Possible columns are:')
            print (meta_data_df.columns)
            return False
        
        #parse the operator
        if operator == 'isin':
            #parse the list of possible values
            values = value.strip('][').split(',')
            values = [adjust_value_type(value) for value in values]
            return (meta_data_df[arg].isin(values))
        else:
            value = adjust_value_type(value)
            if (operator == '>') or (operator.lower() == 'gt'):
                return (meta_data_df[arg] > value)
            elif (operator == '<') or (operator.lower() == 'lt'):
                return (meta_data_df[arg] < value)
            elif (operator == '==') or (operator == '=') or (operator.lower() == 'eq'):
                return (meta_data_df[arg] == value)
            else:
                print(f'Unknown operator: {operator}')
    
    def preprocess_filters(filters):
        condition = True
        for triple in filters:
            condition_from_triple = preprocess_triple(triple)
            condition = condition & condition_from_triple
        return condition  
    
    def stringify_filters(filters):
        #'arg0_op_arg1__arg0_op_arg1'
        return '__'.join(['_'.join([str(x) for x in triple]) for triple in filters]) 
    
    #this one helps us join on path names by stripping the .wav/.json info in the end
    def preprocess_index(s):
        last_two_parts_of_the_path = '/'.join(s.split('/')[-2:])
        last_two_parts_of_the_path_without_file_extension = last_two_parts_of_the_path.split('.')[0]
        common_path = last_two_parts_of_the_path_without_file_extension.strip('_annot')
        return common_path
    
    if features.lower() == 'functional':
        features_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/too_big_for_git/preprocess/ALC_features_Functional.json'
    elif features.lower() == 'lld':
        features_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/too_big_for_git/preprocess/ALC_features_LLD.json'

    features_df = pd.read_json(features_path, orient='index')
    features_df['common_path'] = features_df.index.map(preprocess_index)
    features_df.set_index('common_path', inplace=True)
    
    #absolute path so that we're not worried about where to run the script from
    meta_data_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/intoxicat/data/meta_data_annotation_all_features_130623.json'
    
    meta_data_df = pd.read_json(meta_data_path, orient='index')
    meta_data_df['common_path'] = meta_data_df.index.map(preprocess_index)
    meta_data_df.set_index('common_path', inplace=True)

    condition = preprocess_filters(filters)
    filtered_df = meta_data_df[condition].join(features_df, how='left', on='common_path', lsuffix='_l')
    
    if not preserve_meta_data:
        columns_to_preserve = features_df.columns
        filtered_df = filtered_df[columns_to_preserve]
    
    if balance_classes:
        n_sober = len(filtered_df[filtered_df['intoxicated'].isin(['na','cna'])])
        n_intoxicated = len(filtered_df[filtered_df['intoxicated']=='a'])
        n_to_keep = min(n_sober, n_intoxicated)
        if max_samples: #total max number of rows in the output dataset
            n_to_keep = min(int(max_samples/2), n_to_keep) 
        #print(f'Balance classes: {n_to_keep} samples will be kept for each class')
        filtered_sober_df = filtered_df[filtered_df['intoxicated'].isin(['na','cna'])].sample(n=n_to_keep)
        #print(f'{len(filtered_sober_df)} samples kept for sober class')
        filtered_intoxicated_df = filtered_df[filtered_df['intoxicated']=='a'].sample(n=n_to_keep)
        #print(f'{len(filtered_sober_df)} samples kept for intoxicated class')
        balanced_filtered_df = pd.concat([filtered_sober_df, filtered_intoxicated_df])
        #print(f'{len(balanced_filtered_df)} samples kept overall')
        filtered_df = balanced_filtered_df.sample(frac=1) #shuffle the rows
        print(f'{len(filtered_df)} samples kept overall')
    elif max_samples:
        filtered_df = filtered_df.sample(n=max_samples)
    
    if save_df:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print(f'Created directory: {save_path}')
        out_filename = os.path.join(save_path, f'filtered_{features}_features_{stringify_filters(filters)}_{flags}.json')
        filtered_df.to_json(out_filename, orient='index')
        print(f'Filtered dataframe saved to: {out_filename}')
        #set 744 permission with sticky bit (stat.S_ISVTX sticky bit)
        os.chmod(out_filename, stat.S_IRUSR | stat.S_IWUSR| stat.S_IXUSR| stat.S_IRGRP | stat.S_IROTH | stat.S_ISVTX) 
        
    if return_df:
        return filtered_df

# In[ ]:


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a subset from the data based on filters')

    parser.add_argument('filters', type=str, nargs='*', help='Specify triples of filters in the format: arg1,operator1,value1 arg2,operator2,value2')
    parser.add_argument('out_path', default='preprocess/', type=str)
    parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
    parser.add_argument('--balance_classes', type=bool, default=True, help='If true, datapoints that exceed the number of datapoints in the smallest class will be disregarded (data loss); if false, all retrieved data is preserved (unbalanced classes)')
    parser.add_argument('--preserve_metadata', type=bool, default=False, help='If true, all columns from the metadata json will also be in the output json. Otherwise, only audio features are saved')
    parser.add_argument('--max_samples', type=int, default=None, help='max total rows of both classes in the output data')
    args = vars(parser.parse_args())
    create_subset(filters=[x.split(',') for x in args['filters']], 
                      save_path=args['out_path'], 
                      save_df=True, 
                      return_df=False, 
                      preserve_meta_data=args['preserve_metadata'],
                      features=args['features'],
                      balance_classes=args['balance_classes'],
                      max_samples=args['max_samples']
                 )
