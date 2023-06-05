import numpy as numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# How intoxication is labeled 
ground_truth_labels = ["a", "na"]
predicted_labels = ["a", "na"]

# what I wanted to do here is checking everytime there's "alc" assign "a" as intox and "na" as not_intox
# does it make sense? 

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
