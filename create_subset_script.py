#!/usr/bin/env python
# coding: utf-8

# In[1]:


def create_subset(filters, 
                  path='preprocess/', 
                  save_df=True, 
                  return_df=False, 
                  preserve_meta_data=False,
                  features='Functional'
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
        arg0, operator, arg1 = triple
        if operator == '>':
            return (meta_data_df[arg0] > arg1)
        if operator == '<':
            return (meta_data_df[arg0] < arg1)
        if operator == '==':
            return (meta_data_df[arg0] == arg1)
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
        features_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/preprocess/ALC_features_Functional.json'
    elif features.lower() == 'lld':
        features_path = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/zeidler/preprocess/ALC_features_LLD.json'

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

    parser.add_argument('filters', type=str, nargs='+', help='Specify triples of filters in the format: arg1,operator1,value1 arg2,operator2,value2')
    parser.add_argument('out_path', default='preprocess/', type=str)
    parser.add_argument('preserve_metadata', type=bool, default=False, help='If true, all columns from the metadata json will also be in the output json. Otherwise, only audio features are saved')
    parser.add_argument('features', choices=['Functional', 'LLD'], default='Functional', help='Specify one of: Functional, LLD')
    args = vars(parser.parse_args())
    create_subset(filters=[x.split(',') for x in args['filters']], 
                      path=args['out_path'], 
                      save_df=True, 
                      return_df=False, 
                      preserve_meta_data=args['preserve_metadata'],
                      features=args['features'])