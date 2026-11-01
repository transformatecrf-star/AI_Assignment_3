"""
Task 3 Deliverable: Pure-Python Metric Computation & Slice Analysis Engine
Computes multi-class CM, One-vs-Rest CM, Precision, Recall, Macro F1, Weighted F1, 
and Categorical Log Loss from scratch without sklearn.metrics helper functions.
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def compute_multiclass_confusion_matrix(y_true, y_pred, classes):
    """Computes a K x K multi-class confusion matrix from scratch."""
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t]][class_to_idx[p]] += 1
    return cm


def compute_ovr_confusion_matrices(cm, classes):
    """Derives One-vs-Rest TP, FP, FN, TN for each class from the multi-class CM."""
    ovr_dict = {}
    total = cm.sum()
    for idx, cls in enumerate(classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - (tp + fp + fn)
        ovr_dict[cls] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}
    return ovr_dict


def calculate_metrics_from_scratch(df, classes):
    """Calculates summary and per-class metrics purely from scratch."""
    y_true = df['true_label'].tolist()
    y_pred = df['recommended_department'].tolist()
    scores_raw = [json.loads(s) for s in df['class_scores']]
    
    # 1. Confusion Matrices
    cm = compute_multiclass_confusion_matrix(y_true, y_pred, classes)
    ovr = compute_ovr_confusion_matrices(cm, classes)
    
    # 2. Per-class Precision, Recall, F1 & Macro/Weighted Aggregations
    per_class = {}
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0
    total_samples = len(y_true)
    
    for cls in classes:
        tp = ovr[cls]['TP']
        fp = ovr[cls]['FP']
        fn = ovr[cls]['FN']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn
        
        per_class[cls] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': support}
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    macro_f1 = macro_f1_sum / len(classes)
    weighted_f1 = weighted_f1_sum / total_samples if total_samples > 0 else 0.0
    
    # 3. Categorical Log Loss Computation from scratch
    eps = 1e-15
    log_loss_sum = 0.0
    for true_lbl, score_dict in zip(y_true, scores_raw):
        prob = score_dict.get(true_lbl, eps)
        prob = max(eps, min(1 - eps, prob))  # Clip probabilities for stability
        log_loss_sum += -math.log(prob)
    log_loss = log_loss_sum / total_samples if total_samples > 0 else 0.0

    return {
        'multiclass_cm': cm,
        'ovr_cm': ovr,
        'per_class': per_class,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'log_loss': log_loss
    }


def find_file(filename_candidates, search_dirs):
    """Helper to find a file across multiple possible names and directories."""
    for d in search_dirs:
        for name in filename_candidates:
            p = d / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find any of {filename_candidates} in search directories: {search_dirs}")


def main():
    script_dir = Path(__file__).resolve().parent  # points to Task3/
    root_dir = script_dir.parent                  # points to AI_Eng_Assignment_3/
    search_paths = [root_dir, script_dir, root_dir / "Task2", root_dir / "data"]

    # 1. Load Task 2 validation predictions
    val_preds_path = find_file(["validation_predictions.csv"], search_paths)
    val_preds = pd.read_csv(val_preds_path)
    
    # 2. Load source dataset for text body length
    ticket_candidates = ["support_tickets.csv", "IT Support Ticket Data.csv", "cleaned_support_tickets.csv"]
    ticket_path = find_file(ticket_candidates, search_paths)
    source_tickets = pd.read_csv(ticket_path)
    source_tickets.columns = [str(c).strip().lower() for c in source_tickets.columns]
    
    # Standardize ticket ID column name for merging (Cleaned up loop)
    if 'ticket_id' not in source_tickets.columns:
        for candidate in ['id', 'ticketid', 'unnamed: 0', 'index']:
            if candidate in source_tickets.columns:
                source_tickets.rename(columns={candidate: 'ticket_id'}, inplace=True)
                break
        else:
            source_tickets['ticket_id'] = source_tickets.index.astype(str)
            
    source_tickets['ticket_id'] = source_tickets['ticket_id'].astype(str).str.strip()
    val_preds['ticket_id'] = val_preds['ticket_id'].astype(str).str.strip()

    # Find body column variant
    body_col = 'body'
    for col in source_tickets.columns:
        if col in ['body', 'text', 'content']:
            body_col = col
            break

    # Merge to compute character length for slicing
    merged = pd.merge(val_preds, source_tickets[['ticket_id', body_col]], on='ticket_id', how='left')
    merged['body_length'] = merged[body_col].fillna('').apply(len)
    
    # Define Body-Length Slices using quantiles (Short, Medium, Long)
    q33, q66 = merged['body_length'].quantile([0.33, 0.66])
    
    def slice_len(l):
        if l <= q33:
            return 'Short'
        elif l <= q66:
            return 'Medium'
        return 'Long'

    merged['length_slice'] = merged['body_length'].apply(slice_len)
    classes = sorted(merged['true_label'].unique().tolist())
    
    print("================ OVERALL VALIDATION METRICS ================")
    overall = calculate_metrics_from_scratch(merged, classes)
    print(f"Macro F1-Score:      {overall['macro_f1']:.4f}")
    print(f"Weighted F1-Score:   {overall['weighted_f1']:.4f}")
    print(f"Categorical Log Loss: {overall['log_loss']:.4f}\n")
    
    print("================ BODY-LENGTH SLICE METRICS ================")
    for s_name in ['Short', 'Medium', 'Long']:
        sub_df = merged[merged['length_slice'] == s_name]
        if len(sub_df) > 0:
            res = calculate_metrics_from_scratch(sub_df, classes)
            print(f"Slice: {s_name:<6} | Count: {len(sub_df):<4} | Macro F1: {res['macro_f1']:.4f} | Log Loss: {res['log_loss']:.4f}")
        else:
            print(f"Slice: {s_name:<6} | Count: 0    | No samples available")

if __name__ == "__main__":
    main()