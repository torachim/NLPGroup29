import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

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

# --- Voting Logic (Only for Test Set) ---
def resolve_evasion_label(row):
    # gather votes
    votes = [
        str(row.get('annotator1', '')).strip(),
        str(row.get('annotator2', '')).strip(),
        str(row.get('annotator3', '')).strip()
    ]
    # filter bad values
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None']
    
    if not votes:
        return "Explicit" # Fallback

    counts = Counter(votes)
    most_common = counts.most_common()
    
    # 1. Majority Vote (>=2)
    if most_common[0][1] >= 2:
        return most_common[0][0]
    
    # 2. Clarity Parent Vote
    # if 3 different labels, check if 2 belong to same Clarity class
    parents = [TAXONOMY_MAP.get(v, 'Unknown') for v in votes]
    parent_counts = Counter(parents)
    best_parent = parent_counts.most_common(1)[0][0]
    
    if parent_counts[best_parent] >= 2:
        # take the first label that matches this parent
        for v in votes:
            if TAXONOMY_MAP.get(v) == best_parent:
                return v
                
    # 3. Fallback: Primary Annotator
    return votes[0]

def map_predictions(evasion_preds):
    return [TAXONOMY_MAP.get(label, "Ambivalent") for label in evasion_preds]

def main():
    print("--- Loading Data ---")
    train_df = pd.read_csv("data/processed/train.csv").fillna("")
    test_df = pd.read_csv("data/processed/test.csv").fillna("")
    
    # --- Prepare Training Data ---
    # Use existing 'evasion_label'
    print(f"Training on {len(train_df)} samples (using provided labels)...")
    X_train = train_df['clean_answer']
    y_train_evasion = train_df['evasion_label']
    
    # Filter train data to ensure valid labels
    mask = y_train_evasion.isin(TAXONOMY_MAP.keys())
    X_train = X_train[mask]
    y_train_evasion = y_train_evasion[mask]
    
    # creating clarity labels from evasion labels for direct model
    y_train_clarity = [TAXONOMY_MAP[l] for l in y_train_evasion]

    # --- Prepare Test Data ---
    # Apply Voting Logic to Annotators
    print(f"Testing on {len(test_df)} samples (applying voting logic)...")
    test_df['final_evasion'] = test_df.apply(resolve_evasion_label, axis=1)
    
    X_test = test_df['clean_answer']
    y_test_evasion = test_df['final_evasion']
    y_test_clarity = [TAXONOMY_MAP.get(l, "Ambivalent") for l in y_test_evasion]

    # --- 1. Direct Baseline ---
    print("\n[Baseline 1] Direct Logistic Regression (3-class)...")
    direct_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    
    direct_pipe.fit(X_train, y_train_clarity)
    direct_preds = direct_pipe.predict(X_test)
    
    f1_direct = f1_score(y_test_clarity, direct_preds, average='macro')
    print(f"Direct Macro F1: {f1_direct:.4f}")
    
    # --- 2. Hierarchical Baseline ---
    print("\n[Baseline 2] Hierarchical Logistic Regression (9-class)...")
    hier_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    
    hier_pipe.fit(X_train, y_train_evasion)
    evasion_preds = hier_pipe.predict(X_test)
    mapped_preds = map_predictions(evasion_preds)
    
    f1_hier = f1_score(y_test_clarity, mapped_preds, average='macro')
    print(f"Hierarchical Macro F1: {f1_hier:.4f}")
    
    # Compare
    print("\n=== Final Results ===")
    print(f"Direct:       {f1_direct:.4f}")
    print(f"Hierarchical: {f1_hier:.4f}")

if __name__ == "__main__":
    main()