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

sys.path.append(os.getcwd())

# Import reduced map
from src.dataset import get_datasets, REDUCED_EVASION_MAP, CLARITY_MAP
from src.model import DualHeadXLNet

# --- Config ---
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_reduced_k5.pt" # save as separate model

def get_class_weights(df, label_col, mapping):
    # map first to get integer labels
    labels = df[label_col].map(mapping).values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def train_epoch(model, loader, optimizer, scheduler, loss_fn_c, loss_fn_e):
    model.train()
    total_loss = 0
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        c_label = batch['clarity_labels'].to(DEVICE)
        e_label = batch['evasion_labels'].to(DEVICE)
        
        optimizer.zero_grad()
        c_logits, e_logits = model(ids, mask)
        
        loss_c = loss_fn_c(c_logits, c_label)
        loss_e = loss_fn_e(e_logits, e_label)
        loss = (0.3 * loss_c) + (0.7 * loss_e)
        
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
            
            all_c_pred.extend(torch.argmax(c_logits, dim=1).cpu().numpy())
            all_e_pred.extend(torch.argmax(e_logits, dim=1).cpu().numpy())
            all_c_true.extend(batch['clarity_labels'].numpy())
            all_e_true.extend(batch['evasion_labels'].numpy())
            
    f1_c = f1_score(all_c_true, all_c_pred, average='macro')
    f1_e = f1_score(all_e_true, all_e_pred, average='macro')
    return f1_c, f1_e

def main():
    print(f"--- Starting Reduced Granularity Training (k=5) ---")
    os.makedirs("models", exist_ok=True)
    
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    # LOAD WITH REDUCED FLAG
    train_ds, test_ds = get_datasets(
        "data/processed/train.csv",
        "data/processed/test.csv",
        tokenizer,
        use_reduced=True  # <--- CRITICAL CHANGE
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Calculate weights for REDUCED labels
    train_df = pd.read_csv("data/processed/train.csv")
    # need to map train labels to reduced first for weight calc
    # we simulate the mapping logic from dataset.py here quickly
    # REDUCED_EVASION_MAP maps "Explicit" -> 0, etc.
    w_clarity = get_class_weights(train_df, 'clarity_label', CLARITY_MAP)
    
    # custom helper to get weights for the string column based on the map
    # since column has "Explicit", "Dodging" etc.
    # we map them via REDUCED_EVASION_MAP
    labels = train_df['evasion_label'].map(REDUCED_EVASION_MAP).values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    w_evasion = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    
    print(f"Reduced Evasion Weights (Size {len(w_evasion)}): {w_evasion}")
    
    # Initialize model with 5 labels
    model = DualHeadXLNet(num_evasion_labels=5).to(DEVICE)
    
    criterion_c = nn.CrossEntropyLoss(weight=w_clarity)
    criterion_e = nn.CrossEntropyLoss(weight=w_evasion)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)
    
    best_f1 = 0
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion_c, criterion_e)
        f1_c, f1_e = eval_model(model, test_loader)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Clarity F1: {f1_c:.4f} | Evasion F1 (k=5): {f1_e:.4f}")
        
        if f1_c > best_f1:
            print(f"Saving new best k=5 model...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_f1 = f1_c

    print(f"Done. Best Clarity F1 with Reduced Taxonomy: {best_f1:.4f}")

if __name__ == "__main__":
    main()