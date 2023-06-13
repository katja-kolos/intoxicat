#!/usr/bin/env python
# coding: utf-8

'''
A script that looks at metadata and returns datasamples (as features) that satisfy your filters on metadata.
Assumption: when two or more different tuples of filters are passed the condition is considered as 'and' 
'''


def create_subset(filters, 
                  path='preprocess/', 
                  save_df=True, 
                  return_df=False, 
                  preserve_meta_data=False,
                  features='Functional',
                  balance_classes=True
                 ):
    import json
    import pandas as pd
    import numpy as np
    import os
    #receives filters as [('age', '>', 30), ('sex', '==', 'F')]
    #creates a json file in intoxicat/preprocess
    
    print(filters)
    
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
            if operator == '>':
                return (meta_data_df[arg] > value)
            elif operator == '<':
                return (meta_data_df[arg] < value)
            elif operator == '==':
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
        s = s.split('.')[0]
        return '/'.join(s.split('/')[:-1]) + '/' + '_'.join(s.split('/')[-1].split('_')[:3])
    ###
    
    
    if features.lower() == 'functional':
        features_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/too_big_for_git/preprocess/ALC_features_Functional.json'
    elif features.lower() == 'lld':
        features_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/too_big_for_git/preprocess/ALC_features_LLD.json'

    features_df = pd.read_json(features_path, orient='index')
    features_df['common_path'] = features_df.index.map(preprocess_index)
    features_df.set_index('common_path', inplace=True)
    
    #absolute path so that we're not worried about where to run the script from
    meta_data_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/intoxicat/data/meta_data_annotation_all_features_050623.json'
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
        #print(f'Balance classes: {n_to_keep} samples will be kept for each class')
        filtered_sober_df = filtered_df[filtered_df['intoxicated'].isin(['na','cna'])].sample(n=n_to_keep)
        #print(f'{len(filtered_sober_df)} samples kept for sober class')
        filtered_intoxicated_df = filtered_df[filtered_df['intoxicated']=='a'].sample(n=n_to_keep)
        #print(f'{len(filtered_sober_df)} samples kept for intoxicated class')
        balanced_filtered_df = filtered_sober_df.append(filtered_intoxicated_df)
        #print(f'{len(balanced_filtered_df)} samples kept overall')
        filtered_df = balanced_filtered_df.sample(frac=1) #shuffle the rows
        print(f'{len(filtered_df)} samples kept overall')
    
    if save_df:
        out_path = os.path.join(path, f'filtered_features_{stringify_filters(filters)}.json')
        print(f'Filtered dataframe saved to: {out_path}')
        filtered_df.to_json(out_path, orient='index')
        
    if return_df:
        return filtered_df

# In[ ]:


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a subset from the data based on filters')

    parser.add_argument('filters', type=str, nargs='*', help='Specify triples of filters in the format: arg1,operator1,value1 arg2,operator2,value2')
    parser.add_argument('out_path', default='preprocess/', type=str)
    parser.add_argument('preserve_metadata', type=bool, default=False, help='If true, all columns from the metadata json will also be in the output json. Otherwise, only audio features are saved')
    parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
    parser.add_argument('balance_classes', type=bool, default=True, help='If true, datapoints that exceed the number of datapoints in the smallest class will be disregarded (data loss); if false, all retrieved data is preserved (unbalanced classes)')
    args = vars(parser.parse_args())
    create_subset(filters=[x.split(',') for x in args['filters']], 
                      path=args['out_path'], 
                      save_df=True, 
                      return_df=False, 
                      preserve_meta_data=args['preserve_metadata'],
                      features=args['features'],
                      balance_classes=args['balance_classes']
                 )