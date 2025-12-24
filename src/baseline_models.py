import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.getcwd())
from src.evaluation import get_detailed_metrics

TAXONOMY_MAP = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent", "General": "Ambivalent", "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent", "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply", "Claims ignorance": "Clear Non-Reply", "Clarification": "Clear Non-Reply"
}

def resolve_evasion_label(row):
    votes = [str(row.get(f'annotator{i}', '')).strip() for i in range(1, 4)]
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None' and v != '']
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
    return [TAXONOMY_MAP.get(label, "Ambivalent") for label in evasion_preds]

def main():
    print("--- Loading Data ---")
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    
    if not os.path.exists(train_path):
        print(f"ERROR: File not found at {train_path}.")
        return

    train_df = pd.read_csv(train_path).fillna("")
    test_df = pd.read_csv(test_path).fillna("")
    
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    # Filter valid
    mask = train_df['evasion_label'].isin(TAXONOMY_MAP.keys())
    train_df = train_df[mask]
    
    # --- ADAPTATION: Use 'interview_answer' ---
    X_train = train_df['interview_answer']
    y_train_evasion = train_df['evasion_label']
    y_train_clarity = train_df['clarity_label']

    test_df['final_evasion'] = test_df.apply(resolve_evasion_label, axis=1)
    X_test = test_df['interview_answer']
    y_test_evasion = test_df['final_evasion']
    y_test_clarity = test_df['clarity_label']

    # 1. Direct Baseline
    print("\n[Baseline 1] Direct Logistic Regression (3-class)")
    direct_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    direct_pipe.fit(X_train, y_train_clarity)
    direct_preds = direct_pipe.predict(X_test)
    metrics_direct, report_direct = get_detailed_metrics(y_test_clarity, direct_preds, prefix="Direct_")
    print(report_direct)

    # 2. Hierarchical Baseline
    print("\n[Baseline 2] Hierarchical Logistic Regression")
    hier_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    hier_pipe.fit(X_train, y_train_evasion)
    raw_evasion_preds = hier_pipe.predict(X_test)
    mapped_clarity_preds = map_predictions(raw_evasion_preds)
    
    print(">>> Mapped Clarity Performance")
    metrics_hier, report_hier = get_detailed_metrics(y_test_clarity, mapped_clarity_preds, prefix="Hier_")
    print(report_hier)

    print(">>> Raw Evasion Performance")
    _, report_raw = get_detailed_metrics(y_test_evasion, raw_evasion_preds)
    print(report_raw)

if __name__ == "__main__":
    main()