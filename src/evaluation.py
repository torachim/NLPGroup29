import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def get_detailed_metrics(y_true, y_pred, label_names=None, prefix=""):
    """
    Calculates detailed metrics: F1, Precision, Recall (Macro/Micro), Accuracy 
    and per-class metrics. Returns a dictionary and a printable report string.
    """
    metrics = {}
    
    # Global Metrics
    metrics[f'{prefix}Accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'{prefix}Macro_F1'] = f1_score(y_true, y_pred, average='macro')
    metrics[f'{prefix}Micro_F1'] = f1_score(y_true, y_pred, average='micro')
    metrics[f'{prefix}Macro_Prec'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}Macro_Rec'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Detailed Report (Per Label)
    report_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    
    return metrics, report_str