import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
# imports pytorch and transformers essentials
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.getcwd())
from src.dataset import get_datasets, CLARITY_MAP
from src.model import SingleHeadXLNet
from src.evaluation import get_detailed_metrics

# configuration for batch size learning rate and epochs
BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_direct_clarity.pt"

# runs one full epoch of training
def train_epoch(model, loader, optimizer, scheduler, criterion):
    # sets model to training mode
    model.train()
    total_loss = 0
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE) # target labels for current batch
        
        # clears accumulated gradients
        optimizer.zero_grad()
        # computes model output logits
        logits = model(ids, mask)
        # calculates loss value
        loss = criterion(logits, labels)
        
        # performs backpropagation for gradients
        loss.backward()
        # updates model parameters
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# main function for orchestration
def main():
    print("=== STARTING DIRECT CLARITY TRAINING (3 Classes) ===")
    # creates model directory if needed
    os.makedirs("models", exist_ok=True)
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    # prepares train and test splits with specific mode
    train_ds, test_ds, train_df, _ = get_datasets(
        "data/raw/train.csv", 
        "data/raw/test.csv", 
        tokenizer, 
        mode='clarity'
    )
    # initializes dataloaders for batch processing
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # handle class imbalance
    # assigns higher weights to rare classes for balance
    labels = [CLARITY_MAP[l] for l in train_df['clarity_label']]
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    # computes weights inverse to class frequency
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Clarity Class Weights: {class_weights}")

    # initializes model optimizer and training infrastructure
    model = SingleHeadXLNet(num_labels=3).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)
    
    best_f1 = 0
    target_names = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    
    # main loop managing training and evaluation steps
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # executes single training pass updating weights
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        print(f"  Training Loss: {train_loss:.4f}")
        
        # predicts on test set without learning gradients
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                
                # gets model prediction for batch
                logits = model(ids, mask)
                # converts logits to class indices
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(batch['labels'].numpy()) # clarity labels
        
        # generates detailed classification report and metrics
        metrics, report = get_detailed_metrics(trues, preds, label_names=target_names)
        print("  >>> Evaluation Report (Direct Clarity):")
        print(report)
        
        # saves best model checkpoint if f1 improves
        curr_f1 = metrics['Macro_F1']
        if curr_f1 > best_f1:
            print(f"  [+] New Best Model (F1: {curr_f1:.4f}) -> Saving...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_f1 = curr_f1

    print(f"\nDone. Best Direct Clarity F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()