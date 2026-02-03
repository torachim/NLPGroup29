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
from src.dataset import get_datasets, EVASION_MAP_5
from src.model import SingleHeadXLNet
from src.evaluation import get_detailed_metrics

BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/xlnet_reduced_k5.pt"

# maps 5 reduced classes to clarity taxonomy
# 0:Exp->0(Clear), 1:Active->1(Amb), 2:Vague->1(Amb), 3:Partial->1(Amb), 4:Refusal->2(CNR)
MAPPING_ARR_K5 = np.array([0, 1, 1, 2, 2])

# main orchestration function
def main():
    print("=== STARTING REDUCED TRAINING (k=5) ===")
    os.makedirs("models", exist_ok=True)
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    # loads 5 class dataset and prepares loaders
    train_ds, test_ds, train_df, _ = get_datasets("data/raw/train.csv", "data/raw/test.csv", tokenizer, mode='evasion_5')
    # creates dataloaders for batching
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # ensures balanced learning with 5 class weights
    # map training strings to integers first
    labels_str = train_df['final_evasion_str']
    labels = [EVASION_MAP_5[l] for l in labels_str]
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Reduced Evasion Weights (5 classes): {class_weights}")

    # initializes 5 output model with linear schedule
    model = SingleHeadXLNet(num_labels=5).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*EPOCHS)
    
    best_mapped_f1 = 0
    clarity_names = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # standard training loop updating weights via loss
        # switches to training mode
        model.train()
        train_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE) # reduced evasion target labels
            
            # resets gradient buffer
            optimizer.zero_grad()
            # calculates logits from model
            logits = model(ids, mask)
            # computes loss value
            loss = criterion(logits, labels)
            # computes gradients
            loss.backward()
            # updates model parameters
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        print(f"  Training Loss: {train_loss/len(train_loader):.4f}")

        # collects analysis predictions without gradient tracking
        model.eval()
        evasion_preds = []
        evasion_trues = []
        clarity_trues = [] 
        
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                # gets model predictions for batch
                logits = model(ids, mask)
                
                # generates 5 class predictions for later mapping
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                evasion_preds.extend(preds)
                evasion_trues.extend(batch['labels'].numpy())
                
                # compares mapped predictions to final reality
                clarity_trues.extend(batch['clarity_truth'].numpy())
        
        # verifies model correctly learns reduced taxonomy
        _, report_raw = get_detailed_metrics(evasion_trues, evasion_preds)
        print("  >>> [Internal] Reduced Evasion Performance (5 Classes):")
        print(report_raw)

        # evaluates accuracy on main task via mapping
        evasion_preds_np = np.array(evasion_preds)
        clarity_preds_mapped = MAPPING_ARR_K5[evasion_preds_np]
        
        metrics_map, report_map = get_detailed_metrics(clarity_trues, clarity_preds_mapped, label_names=clarity_names)
        print("  >>> [Target] Mapped Clarity Performance (Reduced Evasion -> Clarity):")
        print(report_map)
        
        # persists model version if mapped metric improves
        curr_f1 = metrics_map['Macro_F1']
        if curr_f1 > best_mapped_f1:
            print(f"  [+] New Best Reduced Model (F1: {curr_f1:.4f}) -> Saving...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_mapped_f1 = curr_f1

    print(f"\nDone. Best Mapped Clarity F1 (k=5): {best_mapped_f1:.4f}")

if __name__ == "__main__":
    main()