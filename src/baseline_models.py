import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.append(os.getcwd())
from src.evaluation import get_detailed_metrics

# --- Taxonomy Definition ---
TAXONOMY_MAP = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply"
}

CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

# --- Voting Logic ---
def resolve_evasion_label(row):
    votes = [
        str(row.get('annotator1', '')).strip(),
        str(row.get('annotator2', '')).strip(),
        str(row.get('annotator3', '')).strip()
    ]
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None']
    
    if not votes: return "Explicit"

    counts = Counter(votes)
    most_common = counts.most_common()
    
    if most_common[0][1] >= 2: return most_common[0][0]
    
    parents = [TAXONOMY_MAP.get(v, 'Unknown') for v in votes]
    parent_counts = Counter(parents)
    best_parent = parent_counts.most_common(1)[0][0]
    
    if parent_counts[best_parent] >= 2:
        for v in votes:
            if TAXONOMY_MAP.get(v) == best_parent: return v
                
    return votes[0]

def map_predictions(evasion_preds):
    # Maps 9 evasion labels -> 3 clarity labels
    return [TAXONOMY_MAP.get(label, "Ambivalent") for label in evasion_preds]

def main():
    print("--- Loading Data ---")
    train_df = pd.read_csv("data/processed/train.csv").fillna("")
    test_df = pd.read_csv("data/processed/test.csv").fillna("")
    
    # Prepare Train (Use provided label)
    print(f"Training set: {len(train_df)} samples")
    X_train = train_df['clean_answer']
    y_train_evasion = train_df['evasion_label']
    
    mask = y_train_evasion.isin(TAXONOMY_MAP.keys())
    X_train = X_train[mask]
    y_train_evasion = y_train_evasion[mask]
    y_train_clarity = [TAXONOMY_MAP[l] for l in y_train_evasion]

    # Prepare Test (Use Voting Logic)
    print(f"Test set: {len(test_df)} samples (applying voting)...")
    test_df['final_evasion'] = test_df.apply(resolve_evasion_label, axis=1)
    
    X_test = test_df['clean_answer']
    y_test_evasion = test_df['final_evasion']
    y_test_clarity = [TAXONOMY_MAP.get(l, "Ambivalent") for l in y_test_evasion]

    # ---------------------------------------------------------
    # 1. Direct Baseline (Input -> 3 Classes)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("[Baseline 1] Direct Logistic Regression (3-class)")
    print("="*50)
    direct_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    
    direct_pipe.fit(X_train, y_train_clarity)
    direct_preds = direct_pipe.predict(X_test)
    
    metrics_direct, report_direct = get_detailed_metrics(y_test_clarity, direct_preds, prefix="Direct_")
    print(report_direct)
    print(f"Macro F1: {metrics_direct['Direct_Macro_F1']:.4f}")

    # ---------------------------------------------------------
    # 2. Hierarchical Baseline (Input -> 9 Classes -> Mapped to 3)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("[Baseline 2] Hierarchical Logistic Regression")
    print("="*50)
    hier_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    
    # Train on Evasion Labels
    hier_pipe.fit(X_train, y_train_evasion)
    
    # Predict Evasion Labels
    raw_evasion_preds = hier_pipe.predict(X_test)
    
    # Map to Clarity Labels
    mapped_clarity_preds = map_predictions(raw_evasion_preds)
    
    # Eval 2a: Mapped Clarity Performance (The most important one!)
    print(">>> [2a] Mapped Clarity Performance (Evasion -> Clarity)")
    metrics_hier, report_hier = get_detailed_metrics(y_test_clarity, mapped_clarity_preds, prefix="Hier_")
    print(report_hier)
    print(f"Macro F1 (Mapped): {metrics_hier['Hier_Macro_F1']:.4f}")

    # Eval 2b: Raw Evasion Performance (Internal Check)
    print("\n>>> [2b] Raw Evasion Performance (9 classes - Internal Check)")
    _, report_raw = get_detailed_metrics(y_test_evasion, raw_evasion_preds)
    print(report_raw)

    # ---------------------------------------------------------
    # Summary Comparison
    # ---------------------------------------------------------
    print("\n=== FINAL COMPARISON (Clarity Task) ===")
    print(f"Direct Approach F1:       {metrics_direct['Direct_Macro_F1']:.4f}")
    print(f"Hierarchical Approach F1: {metrics_hier['Hier_Macro_F1']:.4f}")
    
    diff = metrics_hier['Hier_Macro_F1'] - metrics_direct['Direct_Macro_F1']
    if diff > 0:
        print(f"RESULT: Hierarchical WINS by +{diff:.4f}")
    else:
        print(f"RESULT: Direct WINS by +{abs(diff):.4f}")

if __name__ == "__main__":
    main()