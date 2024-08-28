# Fine Tuning BERT-base for tourist sentiments analysis

In the second code, the BERT base is used for contextual embedding, and the Linear classifier is used for the classification. 
#####################################################################

The following libraries are used in the implementation to train and test the model. 

import pandas as pd #for loading .csv files
import numpy as np  # used for multi-dimensional arrays

from tqdm.auto import tqdm #To show the progress of the model

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder

***********************************************************

BERT tokenizer is used for tokenization. Torch and torch-lightning libraries are used to develop, train, and test the model. 

The pre-trained models (BERT and BERT tokenizer) are imported from the Transformers library downloaded from Hugging Face.

