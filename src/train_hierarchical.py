import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.getcwd())
from src.dataset import get_datasets, EVASION_MAP_9
from src.model import SingleHeadXLNet
from src.evaluation import get_detailed_metrics

BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_hierarchical_k9.pt"

# Mapping: 9 Evasion IDs -> 3 Clarity IDs
# 0:Exp->0(Clear), 1:Dod->1(Amb), 2:Imp->1(Amb), 3:Gen->1(Amb), 4:Def->1(Amb)
# 5:Decl->2(CNR), 6:Ign->2(CNR), 7:Clar->2(CNR), 8:Part->1(Amb)
MAPPING_ARR_K9 = np.array([0, 1, 1, 1, 1, 2, 2, 2, 1])

def main():
    print("=== STARTING HIERARCHICAL TRAINING (k=9) ===")
    os.makedirs("models", exist_ok=True)
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    # 1. Load Data (Mode='evasion_9')
    train_ds, test_ds, train_df, _ = get_datasets("data/raw/train.csv", "data/raw/test.csv", tokenizer, mode='evasion_9')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 2. Class Imbalance Handling (9 Classes)
    labels = [EVASION_MAP_9[l] for l in train_df['final_evasion_str']]
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Evasion Weights (9 classes): {class_weights}")

    # 3. Model (9 Outputs)
    model = SingleHeadXLNet(num_labels=9).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)
    
    best_mapped_f1 = 0
    clarity_names = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE) # Evasion Labels (0-8)
            
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        print(f"  Training Loss: {train_loss/len(train_loader):.4f}")

        # Eval
        model.eval()
        evasion_preds = []
        evasion_trues = []
        clarity_trues = [] # Ground Truth from 'clarity_label' column
        
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                logits = model(ids, mask)
                
                # Predict 9 classes
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                evasion_preds.extend(preds)
                evasion_trues.extend(batch['labels'].numpy())
                
                # Get Clarity Truth for Mapping Eval
                clarity_trues.extend(batch['clarity_truth'].numpy())
        
        # --- REPORT 1: Raw Evasion (Internal Check) ---
        _, report_raw = get_detailed_metrics(evasion_trues, evasion_preds)
        print("  >>> [Internal] Raw Evasion Performance (9 Classes):")
        print(report_raw)

        # --- REPORT 2: Mapped Clarity (Main Hypothesis) ---
        evasion_preds_np = np.array(evasion_preds)
        clarity_preds_mapped = MAPPING_ARR_K9[evasion_preds_np]
        
        metrics_map, report_map = get_detailed_metrics(clarity_trues, clarity_preds_mapped, label_names=clarity_names)
        print("  >>> [Target] Mapped Clarity Performance (Evasion -> Clarity):")
        print(report_map)
        
        # Save based on MAPPED performance
        curr_f1 = metrics_map['Macro_F1']
        if curr_f1 > best_mapped_f1:
            print(f"  [+] New Best Hierarchical Model (F1: {curr_f1:.4f}) -> Saving...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_mapped_f1 = curr_f1

    print(f"\nDone. Best Mapped Clarity F1 (k=9): {best_mapped_f1:.4f}")

if __name__ == "__main__":
    main()