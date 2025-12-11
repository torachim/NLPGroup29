import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter

# --- Mappings ---
EVASION_MAP = {
    "Explicit": 0, "Dodging": 1, "Implicit": 2, "General": 3, "Deflection": 4,
    "Declining to answer": 5, "Claims ignorance": 6, "Clarification": 7, "Partial/half-answer": 8
}

CLARITY_MAP = {
    "Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2
}

TAXONOMY_PARENTS = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent", "General": "Ambivalent", "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent", "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply", "Claims ignorance": "Clear Non-Reply", "Clarification": "Clear Non-Reply"
}

# --- Voting Logic ---
def resolve_evasion_label(row):
    votes = [str(row.get(f'annotator{i}', '')).strip() for i in range(1, 4)]
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None']
    
    if not votes: return row.get('evasion_label', 'Explicit')

    counts = Counter(votes)
    most_common = counts.most_common()
    
    if most_common[0][1] >= 2: return most_common[0][0]
    
    parents = [TAXONOMY_PARENTS.get(v, 'Unknown') for v in votes]
    parent_counts = Counter(parents)
    best_parent = parent_counts.most_common(1)[0][0]
    
    if parent_counts[best_parent] >= 2:
        for v in votes:
            if TAXONOMY_PARENTS.get(v) == best_parent: return v
            
    return votes[0]

# --- Dataset Class ---
class ClarityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # input
        text = f"{row['question']} {self.tokenizer.sep_token} {row['clean_answer']}"
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # targets
        # we assume the dataframe already has 'final_label_str' computed
        label_str = row['final_label_str']
        
        # map to ids
        e_id = EVASION_MAP.get(label_str, 0)
        # derive clarity from evasion to be consistent
        c_str = TAXONOMY_PARENTS.get(label_str, "Ambivalent")
        c_id = CLARITY_MAP.get(c_str, 1)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'clarity_labels': torch.tensor(c_id, dtype=torch.long),
            'evasion_labels': torch.tensor(e_id, dtype=torch.long)
        }

# --- Loader Helper ---
def get_datasets(train_path, test_path, tokenizer):
    # 1. Load Train
    print("loading train data...")
    train_df = pd.read_csv(train_path).fillna("")
    # For train, we trust the provided label
    train_df['final_label_str'] = train_df['evasion_label']
    # filter invalid
    train_df = train_df[train_df['final_label_str'].isin(EVASION_MAP.keys())]
    
    # 2. Load Test
    print("loading test data (with voting)...")
    test_df = pd.read_csv(test_path).fillna("")
    # For test, we apply voting
    test_df['final_label_str'] = test_df.apply(resolve_evasion_label, axis=1)
    # filter invalid
    test_df = test_df[test_df['final_label_str'].isin(EVASION_MAP.keys())]
    
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    train_ds = ClarityDataset(train_df, tokenizer)
    test_ds = ClarityDataset(test_df, tokenizer)
    
    return train_ds, test_ds