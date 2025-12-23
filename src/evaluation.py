from sklearn.metrics import classification_report, f1_score, accuracy_score

def get_detailed_metrics(y_true, y_pred, label_names=None, prefix=""):
    """
    Berechnet Metriken und gibt sowohl ein Dictionary für das Speichern
    als auch einen schönen String für den Print-Output zurück.
    """
    # 1. Berechne Scores für Logik
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 2. Erstelle detaillierten String-Report
    # Dieser String enthält Precision, Recall, F1 pro Klasse und Averages
    report_str = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    
    # Packe wichtige Metriken in Dict
    metrics = {
        f'{prefix}Macro_F1': f1_macro
    }
    
    return metrics, report_str