import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter

# --- MAPPINGS ---

# 1. Standard Evasion (k=9)
EVASION_MAP_9 = {
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

# 2. Reduced Evasion (k=5)
# Strategy: Group classes so they stay within the same Clarity Parent
EVASION_MAP_5 = {
    "Explicit": 0,                  # -> Clear Reply
    "Dodging": 1, "Deflection": 1,  # -> Ambivalent (Active)
    "Implicit": 2, "General": 2,    # -> Ambivalent (Vague)
    "Partial/half-answer": 3,       # -> Ambivalent (Partial)
    "Declining to answer": 4, "Claims ignorance": 4, "Clarification": 4 # -> Clear Non-Reply
}

# 3. Clarity
CLARITY_MAP = {
    "Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2
}

# Parent Lookup (for Voting & Logic)
TAXONOMY_PARENTS = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent", "General": "Ambivalent", "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent", "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply", "Claims ignorance": "Clear Non-Reply", "Clarification": "Clear Non-Reply"
}

# --- VOTING LOGIC (Test Set only) ---
def resolve_evasion_vote(row):
    # Collect votes
    votes = [str(row.get(f'annotator{i}', '')).strip() for i in range(1, 4)]
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None' and v != '']
    
    if not votes: return "Explicit" # Fallback

    counts = Counter(votes)
    most_common = counts.most_common()
    
    # 1. Majority Vote
    if most_common[0][1] >= 2: return most_common[0][0]
    
    # 2. Parent Majority Vote
    parents = [TAXONOMY_PARENTS.get(v, 'Unknown') for v in votes]
    parent_counts = Counter(parents)
    best_parent = parent_counts.most_common(1)[0][0]
    
    if parent_counts[best_parent] >= 2:
        for v in votes:
            if TAXONOMY_PARENTS.get(v) == best_parent: return v
            
    # 3. Fallback
    return votes[0]

# --- DATASET CLASS ---
class ClarityDataset(Dataset):
    def __init__(self, df, tokenizer, mode='clarity', max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode # Options: 'clarity', 'evasion_9', 'evasion_5'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Clean Text (Handle NaNs)
        q_text = str(row.get('clean_question', ""))
        a_text = str(row.get('clean_answer', ""))
        text = f"{q_text} {self.tokenizer.sep_token} {a_text}"
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        # 1. Determine Training Label (based on mode)
        label_id = 0
        evasion_str = str(row.get('final_evasion_str', 'Explicit')) # Calculated in get_datasets
        
        if self.mode == 'clarity':
            # Trust clarity column mostly, but use parent of evasion if generated
            c_str = str(row.get('clarity_label', 'Ambivalent'))
            label_id = CLARITY_MAP.get(c_str, 1)
            
        elif self.mode == 'evasion_9':
            label_id = EVASION_MAP_9.get(evasion_str, 0)
            
        elif self.mode == 'evasion_5':
            label_id = EVASION_MAP_5.get(evasion_str, 0)

        # 2. Always get True Clarity Label (for Evaluation)
        # Assuming 'clarity_label' is always the ground truth column
        c_truth_str = str(row.get('clarity_label', 'Ambivalent'))
        clarity_truth_id = CLARITY_MAP.get(c_truth_str, 1)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long),
            'clarity_truth': torch.tensor(clarity_truth_id, dtype=torch.long)
        }

# --- LOADER FUNCTION ---
def get_datasets(train_path, test_path, tokenizer, mode='clarity'):
    print(f"--- Loading Datasets (Mode: {mode}) ---")
    
    # 1. LOAD TRAIN (Trust 'evasion_label')
    train_df = pd.read_csv(train_path).fillna("")
    train_df['final_evasion_str'] = train_df['evasion_label']
    
    # Filter valid rows
    train_df = train_df[train_df['final_evasion_str'].isin(EVASION_MAP_9.keys())]

    # 2. LOAD TEST (Apply Voting)
    test_df = pd.read_csv(test_path).fillna("")
    print("Applying Democratic Voting Logic to Test Set...")
    test_df['final_evasion_str'] = test_df.apply(resolve_evasion_vote, axis=1)
    
    # Filter valid rows
    test_df = test_df[test_df['final_evasion_str'].isin(EVASION_MAP_9.keys())]
    
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    train_ds = ClarityDataset(train_df, tokenizer, mode=mode)
    test_ds = ClarityDataset(test_df, tokenizer, mode=mode)
    
    return train_ds, test_ds, train_df, test_df