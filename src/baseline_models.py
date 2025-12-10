import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

# load processed data
print("loading data...")
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# handle potential nan in text columns
# fill with empty string to avoid crashes
train_df['clean_answer'] = train_df['clean_answer'].fillna("")
test_df['clean_answer'] = test_df['clean_answer'].fillna("")

# define input and targets
X_train = train_df['clean_answer']
y_train_clarity = train_df['clarity_label']
y_train_evasion = train_df['evasion_label']

X_test = test_df['clean_answer']
y_test_clarity = test_df['clarity_label']
# evasion label might be empty in test if not fully annotated
# but we need it for the hierarchical training flow check
# actually we only evaluate on clarity for the final comparison

# --- Taxonomy Mapping ---
# defines how 9 evasion labels map to 3 clarity labels
# based on your proposal methodology
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

def map_predictions(evasion_preds):
    # maps a list of evasion predictions to clarity labels
    return [TAXONOMY_MAP.get(label, "Ambivalent") for label in evasion_preds]

# --- Model 1: Direct Classification (Baseline) ---
print("\n--- Training Direct Baseline (3-class) ---")
# using simple tf-idf and logistic regression
# class_weight='balanced' helps with the 59% ambivalent skew
direct_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
])

direct_pipe.fit(X_train, y_train_clarity)
direct_preds = direct_pipe.predict(X_test)

# eval direct
print("Direct Baseline Results:")
print(classification_report(y_test_clarity, direct_preds))
f1_direct = f1_score(y_test_clarity, direct_preds, average='macro')
print(f"Macro F1 (Direct): {f1_direct:.4f}")

# --- Model 2: Hierarchical Classification ---
print("\n--- Training Hierarchical Baseline (9-class -> mapped) ---")
# train on evasion labels first
hierarchical_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
])

# filtering training data
# only train on rows where evasion_label is not null/empty
# check for valid evasion labels
mask = y_train_evasion.isin(TAXONOMY_MAP.keys())
X_train_hier = X_train[mask]
y_train_hier = y_train_evasion[mask]

print(f"training on {len(X_train_hier)} samples with valid evasion labels")
hierarchical_pipe.fit(X_train_hier, y_train_hier)

# predict 9 classes
evasion_preds = hierarchical_pipe.predict(X_test)

# map to 3 classes
mapped_clarity_preds = map_predictions(evasion_preds)

# eval hierarchical
print("Hierarchical Baseline Results:")
print(classification_report(y_test_clarity, mapped_clarity_preds))
f1_hier = f1_score(y_test_clarity, mapped_clarity_preds, average='macro')
print(f"Macro F1 (Hierarchical): {f1_hier:.4f}")

# --- Conclusion ---
print("\n=== Result Summary ===")
print(f"Direct Macro F1:       {f1_direct:.4f}")
print(f"Hierarchical Macro F1: {f1_hier:.4f}")

diff = f1_hier - f1_direct
if diff > 0:
    print(f"SUCCESS: Hierarchical approach improved performance by +{diff:.4f}")
else:
    print(f"INSIGHT: Hierarchical approach lag/neutral by {diff:.4f}")