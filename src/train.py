import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import os
import sys

sys.path.append(os.getcwd())
from src.dataset import get_datasets, EVASION_MAP, CLARITY_MAP
from src.model import DualHeadXLNet
from src.evaluation import get_detailed_metrics

# --- Config ---
BATCH_SIZE = 8
MAX_LEN = 512
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_hierarchical.pt"

# --- Mapping Logic for Tensor IDs ---
# We need to map Evasion ID (0-8) to Clarity ID (0-2) efficiently
# Based on EVASION_MAP and CLARITY_MAP in dataset.py
# Explicit(0)->Clear(0), Dodging(1)->Amb(1), etc.
ID_MAPPING_TENSOR = torch.tensor([
    0, # Explicit -> Clear Reply
    1, # Dodging -> Ambivalent
    1, # Implicit -> Ambivalent
    1, # General -> Ambivalent
    1, # Deflection -> Ambivalent
    2, # Declining -> Clear Non-Reply
    2, # Ignorance -> Clear Non-Reply
    2, # Clarification -> Clear Non-Reply
    1  # Partial -> Ambivalent
]).to(DEVICE)

def get_class_weights(df, label_col, mapping):
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
    
    # Containers
    true_clarity = []
    true_evasion = []
    pred_direct_clarity = []
    pred_raw_evasion = []
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            c_logits, e_logits = model(ids, mask)
            
            # 1. Direct Clarity Prediction
            c_pred = torch.argmax(c_logits, dim=1)
            
            # 2. Raw Evasion Prediction
            e_pred = torch.argmax(e_logits, dim=1)
            
            # Store on CPU
            pred_direct_clarity.extend(c_pred.cpu().numpy())
            pred_raw_evasion.extend(e_pred.cpu().numpy())
            true_clarity.extend(batch['clarity_labels'].numpy())
            true_evasion.extend(batch['evasion_labels'].numpy())

    # --- Metrics Calculation ---
    
    # A. Direct Clarity Metrics
    m_direct, _ = get_detailed_metrics(true_clarity, pred_direct_clarity, prefix="Direct_")
    
    # B. Mapped Clarity Metrics (Crucial!)
    # Map raw evasion preds (0-8) to clarity preds (0-2) using numpy map
    # We can recreate the mapping array on CPU for fast lookup
    map_arr = np.array([0, 1, 1, 1, 1, 2, 2, 2, 1]) # Index corresponds to Evasion ID
    pred_mapped_clarity = map_arr[pred_raw_evasion]
    
    m_mapped, _ = get_detailed_metrics(true_clarity, pred_mapped_clarity, prefix="Mapped_")
    
    # C. Raw Evasion Metrics (Internal)
    m_evasion, _ = get_detailed_metrics(true_evasion, pred_raw_evasion, prefix="Evasion_")
    
    return m_direct, m_mapped, m_evasion

def main():
    print(f"--- Standard XLNet Training (k=9) ---")
    os.makedirs("models", exist_ok=True)
    
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    train_ds, test_ds = get_datasets("data/processed/train.csv", "data/processed/test.csv", tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Weights
    train_df = pd.read_csv("data/processed/train.csv")
    w_clarity = get_class_weights(train_df, 'clarity_label', CLARITY_MAP)
    w_evasion = get_class_weights(train_df, 'evasion_label', EVASION_MAP)
    
    model = DualHeadXLNet().to(DEVICE)
    criterion_c = nn.CrossEntropyLoss(weight=w_clarity)
    criterion_e = nn.CrossEntropyLoss(weight=w_evasion)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)
    
    best_mapped_f1 = 0
    
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion_c, criterion_e)
        
        # Get all 3 types of metrics
        m_dir, m_map, m_raw = eval_model(model, test_loader)
        
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
        print(f"  > Direct Clarity F1: {m_dir['Direct_Macro_F1']:.4f}")
        print(f"  > Mapped Clarity F1: {m_map['Mapped_Macro_F1']:.4f} (Target Metric)")
        print(f"  > Raw Evasion F1:    {m_raw['Evasion_Macro_F1']:.4f}")
        
        # Save based on MAPPED performance (since that's our hypothesis)
        if m_map['Mapped_Macro_F1'] > best_mapped_f1:
            print(f"  * New Best Mapped Performance! Saving...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_mapped_f1 = m_map['Mapped_Macro_F1']

    print(f"Done. Best Mapped Clarity F1: {best_mapped_f1:.4f}")

if __name__ == "__main__":
    main()