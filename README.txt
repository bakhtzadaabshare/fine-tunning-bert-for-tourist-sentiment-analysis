In this research, two separate implementations are done. One for machine learning algorithms and another for BERT. Both code are written in Google colab and run on Google TP4 (google backend)

************************************************************************

In machine learning, the following algorithms are trained on the dataset of sentiment analysis to evaluate the informatization in scenic spots. 

Multinomial Naive Bayes Classifier (MNB)         
Logistic Regression (LR)                       
K-Neighbors Classifier (KNN)                  
Random Forest Classifier (RF)                
Voting Classifier (VC)                        
XGBoost Classifier (XGB)                      
Stacking Classifier (SC)  
   
**********************************************************************

For the implementation of the above algorithms, the following packages are used:
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

********************************************************************

For text vectorization the following library is used: 
from sklearn.feature_extraction.text import TfidfVectorizer

********************************************************************
These are the supporting libraries 

import os
import re
import time
import seaborn as sns
import warnings
import nltk
import numpy as np
import pandas as pd  # import Pandas for processing and data analysis
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc
from scikitplot.metrics import plot_roc_curve as auc_roc 

*********************************************************************

#####################################################################

In the second code, the BERT base is used for contextual embedding and Linear classifier is used for the classification. 
the following libraries are used in the implementation to train and test the model. 

import pandas as pd #for loading .csv files
import numpy as np  #using for multi dimensional arraies

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

The pretrained model (BERT and BERT tokenizer) are imported from the Transformers library.


