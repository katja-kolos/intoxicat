import numpy as numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# How intoxication is labeled 
ground_truth_labels = ["a", "na"]
predicted_labels = ["a", "na"]

# what I wanted to do here is checking everytime there's "alc" assign "a" as intox and "na" as not_intox
# does it make sense? 


# Accuracy
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print("Accuracy: ", accuracy)

# Precision 
precision = precision_score(ground_truth_labels, predicted_labels)
print("Precision: ", precision)

# Recall
recall = recall_score(ground_truth_labels, predicted_labels)
print("Recall: ", recall)

# F1 score
f1 = f1_score(ground_truth_labels, predicted_labels)
print("F1: ", f1)