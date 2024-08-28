# Description: This file is used to load the BERT model and tokenizer for the purpose of fine-tuning the model for the task of text classification
# The BERT model is loaded using the pre-trained model provided by the BERT class and the tokenizer is loaded using the pre-trained tokenizer provided by the BertTokenizer class
# The BERT model and tokenizer are then used for tokenizing the text data and fine-tuning the model for text classification tasks
# The BERT model and tokenizer are initialized and stored in the bert_model and tokenizer variables for further use in the text classification task
# The BERT model is fine-tuned using the pre-trained weights and the tokenizer is used for tokenizing the text data for input to the model
# The BERT model and tokenizer are essential components for text classification tasks using BERT-based models

# import necessary libraries
from transformers import BertTokenizer, BertModel
def loadBertModel():
    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    return tokenizer, bert_model