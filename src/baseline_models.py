import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --- Taxonomy Definition ---
# needed for logic: distinct evasion -> same clarity parent
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

# --- Voting Logic ---
def resolve_evasion_label(row):
    # get votes from 3 annotators
    # fill nan with empty string just in case
    votes = [
        str(row.get('annotator1', '')).strip(),
        str(row.get('annotator2', '')).strip(),
        str(row.get('annotator3', '')).strip()
    ]
    
    # filter out empty votes or 'nan'
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None']
    
    if not votes:
        # fallback if no annotators: use provided label or default
        return row.get('evasion_label', 'Explicit')

    counts = Counter(votes)
    most_common = counts.most_common()
    
    # case 1: majority vote (2 or 3 agree)
    # e.g., [('Dodging', 2), ('General', 1)] -> Dodging
    if most_common[0][1] >= 2:
        return most_common[0][0]
    
    # case 2: 3 different votes (1 vs 1 vs 1)
    # check if they belong to same clarity parent
    parents = [TAXONOMY_MAP.get(v, 'Unknown') for v in votes]
    parent_counts = Counter(parents)
    
    # check if a clarity parent has majority (>=2)
    # e.g. votes: Dodging(Amb), General(Amb), Explicit(Clear)
    # parent counts: Ambivalent: 2, Clear: 1
    # winner: Ambivalent
    best_parent = parent_counts.most_common(0)[0][0] # get winner parent
    
    if parent_counts[best_parent] >= 2:
        # pick the first evasion label that matches this parent
        for v in votes:
            if TAXONOMY_MAP.get(v) == best_parent:
                return v
                
    # case 3: total chaos or fallback
    # just take annotator 1 (primary)
    return votes[0]

def map_predictions(evasion_preds):
    return [TAXONOMY_MAP.get(label, "Ambivalent") for label in evasion_preds]

# --- Main Pipeline ---
def main():
    print("loading training data...")
    # we only use train.csv for this phase
    # we split it internally
    df = pd.read_csv("data/processed/train.csv")
    
    # handle nans in text
    df['clean_answer'] = df['clean_answer'].fillna("")
    
    # apply democratic voting
    print("applying democratic voting logic to annotators...")
    df['final_evasion'] = df.apply(resolve_evasion_label, axis=1)
    
    # filter invalid labels just in case
    df = df[df['final_evasion'].isin(TAXONOMY_MAP.keys())]
    
    print(f"data shape after label resolution: {df.shape}")
    
    # features and targets
    X = df['clean_answer']
    y_clarity = df['clarity_label']
    y_evasion = df['final_evasion']
    
    # split 80/20
    # stratify by evasion label to ensure rare classes are in both sets
    print("splitting 80/20 train/val...")
    X_train, X_val, y_ev_train, y_ev_val = train_test_split(
        X, y_evasion, test_size=0.2, random_state=42, stratify=y_evasion
    )
    
    # recreate clarity labels for split
    # we derive them from the split evasion labels to be safe
    y_cl_train = [TAXONOMY_MAP[l] for l in y_ev_train]
    y_cl_val = [TAXONOMY_MAP[l] for l in y_ev_val]
    
    # --- 1. Direct Baseline (3 classes) ---
    print("\n[Baseline 1] Training Direct Logistic Regression (3-class)...")
    direct_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    
    direct_pipe.fit(X_train, y_cl_train)
    direct_preds = direct_pipe.predict(X_val)
    
    print(">>> Results Direct Baseline (Clarity Level):")
    print(classification_report(y_cl_val, direct_preds))
    f1_direct = f1_score(y_cl_val, direct_preds, average='macro')
    print(f"Macro F1: {f1_direct:.4f}")
    
    # --- 2. Hierarchical Baseline (9 classes -> 3 classes) ---
    print("\n[Baseline 2] Training Hierarchical Logistic Regression (9-class)...")
    hier_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    
    # train on 9 classes
    hier_pipe.fit(X_train, y_ev_train)
    
    # predict 9 classes
    evasion_preds = hier_pipe.predict(X_val)
    
    # map to 3 classes
    mapped_preds = map_predictions(evasion_preds)
    
    print(">>> Results Hierarchical Baseline (Clarity Level):")
    print(classification_report(y_cl_val, mapped_preds))
    f1_hier = f1_score(y_cl_val, mapped_preds, average='macro')
    print(f"Macro F1 (Clarity Mapped): {f1_hier:.4f}")
    
    # extra: evaluate evasion level (internal check)
    print("\n>>> Internal Check: Evasion Level Performance (9-class):")
    print(classification_report(y_ev_val, evasion_preds))
    
    # compare
    print("\n=== Final Comparison ===")
    print(f"Direct F1:       {f1_direct:.4f}")
    print(f"Hierarchical F1: {f1_hier:.4f}")
    
    if f1_hier > f1_direct:
        print("Hypothesis Supported: Hierarchical approach is better.")
    else:
        print("Hypothesis Not Supported (for Baseline): Direct approach is better/equal.")

if __name__ == "__main__":
    main()