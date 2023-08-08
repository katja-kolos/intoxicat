import json, os, random, torch, sys, time, argparse, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import metrics 
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import _LRScheduler

sys.path.append('preprocess/')
from prepare_data import split_dataset_into_splits

sys.path.append('evaluation/')
from intoxicat_evaluation import *


# code from https://github.com/DigitalPhonetics/IMS-Toucan/blob/ToucanTTS/Utility/WarmupScheduler.py
class ToucanWarmupScheduler(_LRScheduler):
    """
    A warmup scheduler that should be called after every batch.
    """

    def __init__(self, optimizer, peak_lr=0.001, warmup_steps=8000, max_steps=100000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.max_steps = max_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if step_num <= self.warmup_steps:
            lr = self.peak_lr * min(step_num / self.warmup_steps, 1.0)
            return [lr for _ in self.base_lrs]
        else:
            scale = 1 - (((step_num - self.warmup_steps) / self.max_steps) / (self.max_steps / 10))
            return [max(lr * scale, 1e-7) for lr in self.base_lrs]


def plot_confusion_matrix(targets, predictions, model_name):

    targets = torch.argmax(targets, dim=1)
    predictions = torch.argmax(predictions, dim=1)

    confusion_matrix = metrics.confusion_matrix(targets.cpu(), predictions.cpu())

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Sober', 'Intoxicated'])

    cm_display.plot()

    file_name = '{}/plots/{}_predictions.png'.format('/'.join(model_name.split('/')[:-1]), model_name.split('/')[-1].strip('.pt'))
    
    plt.savefig(file_name)