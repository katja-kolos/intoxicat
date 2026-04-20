# Intoxicat: A Project for Identifying Intoxicated Speech

This project was created for the course "Computational Linguistics Team Laboratory: Phonetics" at the [IMS](https://www.ims.uni-stuttgart.de/) in Stuttgart.
In this project, we tackled the task of intoxication detection using both a simple neural network and a LSTM network and applying different preprocessing techniques such as global speaker normalization on speech data from LMU's [Alcohol Language Corpus (ALC)](https://www.en.phonetik.uni-muenchen.de/research/completed_projects/alc.html).

This repo contains code for:

- preprocessing data from the ALC dataset
- analysing the data based on given meta data annotations
- building different data subsets based on given meta data-related factors
- extracting Functional and LLD features using the opensmile library
- normalising the extracted features based on the speaker
- training two types of neural models (namely a simple neural network and an LSTM network)
- testing and analysing the results of the models

Authors: Ekaterina Kolos & Laura Zeidler (further including some code from Francesca Carlon)

## Installation of Requirements

Simply install required packages by typing:

```
pip install -r requirements.txt
```

## Training the Models

In order to train the models, one can simply run the bash files `train.sh` and `train_lstm.sh` to train the simple neural network (SNN) and the LSTM model respectively. The instruction for the command that has to be specified can be found in the `train_models.py` script. Results (accuracy) will be written in the files `snn_results.csv` or `lstm_results.csv`, depending on the model architecture.

## Evaluating the Models

The models are already evaluated right after training, however, if one wishes to evaluate the already trained and stored models, they can use either the `evaluate_model.py` script or the `evaluate_models_notebook.ipynb` notebook.
