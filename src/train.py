import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import os
import sys

# allow imports from current dir
sys.path.append(os.getcwd())

from src.dataset import get_datasets, EVASION_MAP, CLARITY_MAP
from src.model import DualHeadXLNet

# --- Config ---
BATCH_SIZE = 4        # reduce if oom error
MAX_LEN = 512         # xlnet handles long seqs
EPOCHS = 1
LR = 2e-5             # learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_hierarchical.pt"

# --- Helper: Calc Class Weights ---
def get_class_weights(df, label_col, mapping):
    # compute weights for imbalance handling
    labels = df[label_col].map(mapping).values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def train_epoch(model, loader, optimizer, scheduler, loss_fn_c, loss_fn_e):
    model.train()
    total_loss = 0
    
    for batch in loader:
        # move to device
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        c_label = batch['clarity_labels'].to(DEVICE)
        e_label = batch['evasion_labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # forward pass
        c_logits, e_logits = model(ids, mask)
        
        # calc individual losses
        loss_c = loss_fn_c(c_logits, c_label)
        loss_e = loss_fn_e(e_logits, e_label)
        
        # weighted sum (70% evasion, 30% clarity)
        # as defined in methodology
        loss = (0.3 * loss_c) + (0.7 * loss_e)
        
        # backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def eval_model(model, loader):
    model.eval()
    all_c_true, all_c_pred = [], []
    all_e_true, all_e_pred = [], []
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            c_logits, e_logits = model(ids, mask)
            
            # get predictions
            c_pred = torch.argmax(c_logits, dim=1).cpu().numpy()
            e_pred = torch.argmax(e_logits, dim=1).cpu().numpy()
            
            all_c_pred.extend(c_pred)
            all_e_pred.extend(e_pred)
            all_c_true.extend(batch['clarity_labels'].numpy())
            all_e_true.extend(batch['evasion_labels'].numpy())
            
    # calc macro f1
    f1_c = f1_score(all_c_true, all_c_pred, average='macro')
    f1_e = f1_score(all_e_true, all_e_pred, average='macro')
    
    return f1_c, f1_e

def main():
    print(f"using device: {DEVICE}")
    os.makedirs("models", exist_ok=True)
    
    # 1. load data
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    train_ds, test_ds = get_datasets(
        "data/processed/train.csv",
        "data/processed/test.csv",
        tokenizer
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 2. setup weights
    # load df just to calc weights
    train_df = pd.read_csv("data/processed/train.csv")
    w_clarity = get_class_weights(train_df, 'clarity_label', CLARITY_MAP)
    w_evasion = get_class_weights(train_df, 'evasion_label', EVASION_MAP)
    
    print(f"Clarity Weights: {w_clarity}")
    print(f"Evasion Weights: {w_evasion}")
    
    # 3. setup model & loss
    model = DualHeadXLNet().to(DEVICE)
    
    # weighted cross entropy
    criterion_c = nn.CrossEntropyLoss(weight=w_clarity)
    criterion_e = nn.CrossEntropyLoss(weight=w_evasion)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 4. training loop
    best_f1 = 0
    
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        # train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion_c, criterion_e)
        
        # validate
        val_f1_c, val_f1_e = eval_model(model, test_loader)
        
        # metric of interest: clarity f1 (primary task)
        # or average of both? let's track clarity per proposal
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val F1 Clarity: {val_f1_c:.4f} | Val F1 Evasion: {val_f1_e:.4f}")
        
        # save best
        if val_f1_c > best_f1:
            print(f"found new best model! saving to {SAVE_PATH}")
            torch.save(model.state_dict(), SAVE_PATH)
            best_f1 = val_f1_c
            
    print(f"\nTraining complete. Best Clarity F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()