import numpy as numpy
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# How intoxication is labeled 
# ground_truth_labels = ["a", "na", "cna"]
# predicted_labels = ["a", "na", "cna"]

def calculate_accuracy(ground_truth_labels, predicted_labels):
  # Accuracy
  accuracy = accuracy_score(ground_truth_labels, predicted_labels)
  print("Accuracy: ", accuracy)
  return accuracy

def calculate_precision(ground_truth_labels, predicted_labels):
  # Precision 
  precision = precision_score(ground_truth_labels, predicted_labels)
  print("Precision: ", precision)
  return precision

def calculate_recall(ground_truth_labels, predicted_labels):
  # Recall
  recall = recall_score(ground_truth_labels, predicted_labels)
  print("Recall: ", recall)
  return recall

def calculate_f1(ground_truth_labels, predicted_labels):
  # F1 score
  f1 = f1_score(ground_truth_labels, predicted_labels)
  print("F1: ", f1)
  return f1

def make_labels_human_readable(alc_results):
  human_read_list = []

  for result in alc_results:

     # Check if the word contains "alc"
      if "alc" in alc_results:

          # Assign the appropriate label based on the word
          if result == "a":
              label = "intoxicated"

          elif result == "na":
              label = "not intoxicated"

          else:
              label = "not intoxicated (c)"

          human_read_list.append(label)
          
  return human_read_list


# --------------------------------------------------------------------------------------------
# author: Laura


def check_acc_for_groups(meta_data_path, predictions_path, filters):

  # helper functions taken from create_subset_script.py

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
  
  #this one helps us join on path names by stripping the .wav/.json info in the end
  def preprocess_index(s):
      last_two_parts_of_the_path = '/'.join(s.split('/')[-2:])
      last_two_parts_of_the_path_without_file_extension = last_two_parts_of_the_path.split('.')[0]
      common_path = last_two_parts_of_the_path_without_file_extension.strip('_annot')
      return common_path

  # taken from create_subset_script.py
  predictions_df = pd.read_json(predictions_path, orient='index')
  predictions_df['common_path'] = predictions_df.index.map(preprocess_index)
  predictions_df.set_index('common_path', inplace=True)
  predictions_df.rename(columns={0:'Targets', 1: 'Predictions'}, inplace=True)  
  print(predictions_df)
  
  meta_data_df = pd.read_json(meta_data_path, orient='index')
  meta_data_df['common_path'] = meta_data_df.index.map(preprocess_index)
  meta_data_df.set_index('common_path', inplace=True)
  # print(meta_data_df.columns)

  condition = preprocess_filters(filters)
  # filtered_df = meta_data_df[condition].join(predictions_df, how='left', on='common_path', lsuffix='_l')
  filtered_df = meta_data_df[condition].join(predictions_df, how='inner', on='common_path', lsuffix='_l')

  print(filtered_df)
  print(filtered_df.columns)
  targets = list(filtered_df['Targets'])
  predictions = list(filtered_df['Predictions'])
  print(len(targets) == len(predictions))

  print(f'Targets: {targets}')
  print(f'Predictions: {predictions}')
  return calculate_accuracy(targets, predictions)
    

# def plot_confusion_matrix(targets, predictions, train_loop=True, model_name=None):
# 
#   if one_hot:
#     targets = torch.argmax(targets, dim=1)
#     predictions = torch.argmax(predictions, dim=1)
# 
#   confusion_matrix = metrics.confusion_matrix(targets.cpu(), predictions.cpu())
# 
#   cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Sober', 'Intoxicated'])
# 
#   cm_display.plot()
# 
#   if train_loop:
#     file_name = model_name.split('.pt')[0] + '_predictions.png'
#     plt.savefig(file_name)
