from sklearn.metrics import classification_report, f1_score, accuracy_score

def get_detailed_metrics(y_true, y_pred, label_names=None, prefix=""):
    
    # returns metrics dict and report string
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    report_str = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    
    metrics = {
        f'{prefix}Macro_F1': f1_macro
    }
    
    return metrics, report_str