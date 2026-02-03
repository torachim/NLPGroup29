import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# imports necessary libraries for data processing and modeling
from sklearn.pipeline import Pipeline
import sys
import os

# adds current directory to path for importing src modules
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.evaluation import get_detailed_metrics

# maps evasion types to clarity relationships
TAXONOMY_MAP = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent", "General": "Ambivalent", "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent", "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply", "Claims ignorance": "Clear Non-Reply", "Clarification": "Clear Non-Reply"
}

TAXONOMY_PARENTS = TAXONOMY_MAP 

# determines final label based on annotator consensus
def resolve_evasion_label(row):
    # collects votes from three annotator columns
    votes = [str(row.get(f'annotator{i}', '')).strip() for i in range(1, 4)]
    # filters out empty or invalid votes
    votes = [v for v in votes if v and v.lower() != 'nan' and v != 'None' and v != '']
    if not votes: return "Explicit"
    counts = Counter(votes)
    most_common = counts.most_common()
    # checks if strict majority exists
    if most_common[0][1] >= 2: return most_common[0][0]
    # looks for agreement on parent category level
    parents = [TAXONOMY_MAP.get(v, 'Unknown') for v in votes]
    parent_counts = Counter(parents)
    best_parent = parent_counts.most_common(1)[0][0]
    if parent_counts[best_parent] >= 2:
        for v in votes:
            if TAXONOMY_MAP.get(v) == best_parent: return v
    return votes[0]

# transforms fine grained evasion predictions to clarity labels
def map_predictions(evasion_preds):
    return [TAXONOMY_MAP.get(label, "Ambivalent") for label in evasion_preds]

def run_baselines(train_path, test_path):
    # executes baselines training and returns results dictionary
    print(f"--- Running Baselines (Train: {train_path}) ---")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")

    # reads csv files into pandas dataframes
    train_df = pd.read_csv(train_path).fillna("")
    test_df = pd.read_csv(test_path).fillna("")
    
    # filters valid evasion labels removing unknown data
    mask = train_df['evasion_label'].isin(TAXONOMY_MAP.keys())
    train_df = train_df[mask]
    
    # extracts training features and target label columns
    X_train = train_df['interview_answer']
    y_train_evasion = train_df['evasion_label']
    y_train_clarity = train_df['clarity_label']

    test_df['final_evasion'] = test_df.apply(resolve_evasion_label, axis=1)
    X_test = test_df['interview_answer']
    y_test_evasion = test_df['final_evasion']
    y_test_clarity = test_df['clarity_label']

    results = {}

    # direct logistic regression predicting clarity from text
    print("Training Direct Logistic Regression...")
    # builds pipeline with tfidf and logistic regression
    direct_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    # fits model on training data and predicts on test
    direct_pipe.fit(X_train, y_train_clarity)
    direct_preds = direct_pipe.predict(X_test)
    
    metrics_direct, report_direct = get_detailed_metrics(y_test_clarity, direct_preds, prefix="Direct_")
    results['Direct'] = {
        'metrics': metrics_direct,
        'report': report_direct,
        'y_true': y_test_clarity,
        'y_pred': direct_preds
    }

    # hierarchical logistic regression predicting evasion then clarity
    print("Training Hierarchical Logistic Regression...")
    # creates hierarchical pipeline for evasion
    hier_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    ])
    hier_pipe.fit(X_train, y_train_evasion)
    # gets raw evasion predictions before mapping
    raw_evasion_preds = hier_pipe.predict(X_test)
    mapped_clarity_preds = map_predictions(raw_evasion_preds)
    
    metrics_hier, report_hier = get_detailed_metrics(y_test_clarity, mapped_clarity_preds, prefix="Hier_")
    results['Hierarchical'] = {
        'metrics': metrics_hier,
        'report': report_hier,
        'y_true': y_test_clarity,
        'y_pred': mapped_clarity_preds
    }
    
    return results

def main():
    # default csv data file paths
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    
    try:
        results = run_baselines(train_path, test_path)
        print("\n=== Direct Results ===")
        print(results['Direct']['report'])
        print("\n=== Hierarchical Results ===")
        print(results['Hierarchical']['report'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()