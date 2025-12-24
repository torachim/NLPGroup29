import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Importiere Module aus aktuellem Verzeichnis
sys.path.append(os.getcwd())
from src.dataset import get_datasets, CLARITY_MAP
from src.model import SingleHeadXLNet
from src.evaluation import get_detailed_metrics

# --- CONFIG ---
BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_direct_clarity.pt"

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE) # Clarity Labels (0-2)
        
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    print("=== STARTING DIRECT CLARITY TRAINING (3 Classes) ===")
    os.makedirs("models", exist_ok=True)
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    # 1. Load Data (Mode='clarity')
    train_ds, test_ds, train_df, _ = get_datasets(
        "data/raw/train.csv", 
        "data/raw/test.csv", 
        tokenizer, 
        mode='clarity'
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 2. Class Imbalance Handling
    # Wir berechnen Gewichte für die 3 Clarity Klassen
    labels = [CLARITY_MAP[l] for l in train_df['clarity_label']]
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Clarity Class Weights: {class_weights}")

    # 3. Model Setup
    model = SingleHeadXLNet(num_labels=3).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)
    
    best_f1 = 0
    target_names = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    
    # 4. Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        print(f"  Training Loss: {train_loss:.4f}")
        
        # Eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                
                logits = model(ids, mask)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(batch['labels'].numpy()) # Das sind hier Clarity Labels
        
        # Detailed Metrics
        metrics, report = get_detailed_metrics(trues, preds, label_names=target_names)
        print("  >>> Evaluation Report (Direct Clarity):")
        print(report)
        
        # Save Best
        curr_f1 = metrics['Macro_F1']
        if curr_f1 > best_f1:
            print(f"  [+] New Best Model (F1: {curr_f1:.4f}) -> Saving...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_f1 = curr_f1

    print(f"\nDone. Best Direct Clarity F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()