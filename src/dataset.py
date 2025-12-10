import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# mapping based on your dataset analysis
# evasion labels (9 classes)
EVASION_MAP = {
    "Explicit": 0,
    "Dodging": 1,
    "Implicit": 2,
    "General": 3,
    "Deflection": 4,
    "Declining to answer": 5,
    "Claims ignorance": 6,
    "Clarification": 7,
    "Partial/half-answer": 8
}

# clarity labels (3 classes)
CLARITY_MAP = {
    "Clear Reply": 0,
    "Ambivalent": 1,
    "Clear Non-Reply": 2
}

class ClarityDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        # load data
        self.df = pd.read_csv(csv_path)
        
        # fill nans to avoid crashes
        self.df['clean_answer'] = self.df['clean_answer'].fillna("")
        self.df['question'] = self.df['question'].fillna("")
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # combine question and answer for context
        # xlnet format: Question <sep> Answer <cls>
        text = f"{row['question']} {self.tokenizer.sep_token} {row['clean_answer']}"
        
        # tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # get label ids
        # safety check for unknown labels (default to -1 or 0)
        clarity_id = CLARITY_MAP.get(row['clarity_label'], 0)
        
        # for evasion, some might be empty/nan in test set?
        # assuming training data is clean from phase 1
        evasion_id = EVASION_MAP.get(row['evasion_label'], 0) 

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'clarity_labels': torch.tensor(clarity_id, dtype=torch.long),
            'evasion_labels': torch.tensor(evasion_id, dtype=torch.long)
        }