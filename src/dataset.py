import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter

# mappings
EVASION_MAP_9 = {
    "Explicit": 0, "Dodging": 1, "Implicit": 2, "General": 3, "Deflection": 4,
    "Declining to answer": 5, "Claims ignorance": 6, "Clarification": 7, "Partial/half-answer": 8
}

EVASION_MAP_5 = {
    "Explicit": 0,
    
    "Dodging": 1, 
    "Deflection": 1,
    
    "Implicit": 2, 
    "General": 2, 
    "Partial/half-answer": 2,
    
    "Declining to answer": 3, 
    "Claims ignorance": 3,
    
    "Clarification": 4
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

# voting logic for disagreements
def resolve_evasion_vote(row):
    votes = [str(row.get(f'annotator{i}', '')).strip() for i in range(1, 4)]
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None' and v != '']
    
    if not votes: return "Explicit"

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

# dataset class
class ClarityDataset(Dataset):
    def __init__(self, df, tokenizer, mode='clarity', max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # combine question and answer
        q_text = str(row.get('question', ""))
        a_text = str(row.get('interview_answer', ""))
        
        text = f"{q_text} {self.tokenizer.sep_token} {a_text}"
        
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        # target label logic
        label_id = 0
        evasion_str = str(row.get('final_evasion_str', 'Explicit'))
        
        if self.mode == 'clarity':
            c_str = str(row.get('clarity_label', 'Ambivalent'))
            label_id = CLARITY_MAP.get(c_str, 1)
        elif self.mode == 'evasion_9':
            label_id = EVASION_MAP_9.get(evasion_str, 0)
        elif self.mode == 'evasion_5':
            label_id = EVASION_MAP_5.get(evasion_str, 0)

        # ground truth for eval
        c_truth_str = str(row.get('clarity_label', 'Ambivalent'))
        clarity_truth_id = CLARITY_MAP.get(c_truth_str, 1)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long),
            'clarity_truth': torch.tensor(clarity_truth_id, dtype=torch.long)
        }

def get_datasets(train_path, test_path, tokenizer, mode='clarity'):
    print(f"--- Loading Datasets (Mode: {mode}) ---")
    
    # load train
    train_df = pd.read_csv(train_path).fillna("")
    train_df['final_evasion_str'] = train_df['evasion_label']
    train_df = train_df[train_df['final_evasion_str'].isin(EVASION_MAP_9.keys())]

    # load test
    test_df = pd.read_csv(test_path).fillna("")
    print("Applying Democratic Voting Logic...")
    test_df['final_evasion_str'] = test_df.apply(resolve_evasion_vote, axis=1)
    test_df = test_df[test_df['final_evasion_str'].isin(EVASION_MAP_9.keys())]
    
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    train_ds = ClarityDataset(train_df, tokenizer, mode=mode)
    test_ds = ClarityDataset(test_df, tokenizer, mode=mode)
    
    return train_ds, test_ds, train_df, test_df