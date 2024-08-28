#importing necessary libraries
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

#Inhereting PyTorch Dataset class for handling toxic comments data accordingly.
class tourismDataset(Dataset):
  def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.selected_text
    labels = data_row['sentiment']

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )