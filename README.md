# Fine Tuning BERT-base for tourist sentiments analysis

In the second code, the BERT base is used for contextual embedding, and the Linear classifier is used for the classification. 
**************************************************************************

The following libraries are used in the implementation to train and test the model. 

1. import pandas as pd #for loading .csv files
2. import numpy as np  # used for multi-dimensional arrays
3. from tqdm.auto import tqdm #To show the progress of the model
4. import torch
5. import torch.nn as nn
6. from torch.utils.data import Dataset, DataLoader
7. from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
8. import pytorch_lightning as pl
9. from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
10. from sklearn.model_selection import train_test_split
11. from sklearn.metrics import classification_report, multilabel_confusion_matrix
12. from sklearn.preprocessing import LabelEncoder

***********************************************************

BERT tokenizer is used for tokenization. Torch and torch-lightning libraries are used to develop, train, and test the model. 

The pre-trained models (BERT and BERT tokenizer) are imported from the Transformers library downloaded from Hugging Face.

# How to train the model on your dataset? 
Before going to use the code install the following libraries 

1. **Transformers**: Can be installed using the command "pip install transformers".
2. **PyTorch**: Can be installed using the command "pip3 install torch torchvision torchaudio" **CPU** or "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
3. **Pytorch Lightning**: Can be installed using the command "pip install pytorch-lightning"

After the installation make only changes in the data_loader.py and trainer.py

