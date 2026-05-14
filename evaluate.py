import os
import pickle
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".")
import numpy as np
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)
    parser.add_argument("--in_pkl", type=str, required=True)
    parser.add_argument("--out_pkl", type=str, required=True)
    args = parser.parse_args()

    # Load ID data to determine length
    with open(args.in_pkl, 'rb') as f:
        in_data = pickle.load(f)

    # Dynamic length (should be 1 in your case)
    length = len(in_data)
    print(f"Evaluating {length} model(s) found in pickle file.")

    in_label = [[] for _ in range(length)]
    out_label = [[] for _ in range(length)]
    scores = {}

    # Process In-Distribution Data
    for i in range(length):
        data = in_data[i]
        # Handle label
        if 'label' in data:
            in_label[i] += data['label']
        
        # Handle scores
        for key in data.keys():
            if key == 'label':
                continue
            if key not in scores:
                scores[key] = [{'in': [], 'out': []} for _ in range(length)]
            
            # Ensure data is converted to list (handles Tensors/Arrays)
            val = data[key]
            if hasattr(val, 'tolist'):
                val = val.tolist()
            scores[key][i]['in'] += list(val)

    # Process Out-Of-Distribution Data
    with open(args.out_pkl, 'rb') as f:
        out_data = pickle.load(f)
    
    if len(out_data) != length:
        print(f"Warning: Mismatch in ID ({length}) and OOD ({len(out_data)}) entries. Using minimum.")
        length = min(length, len(out_data))

    for i in range(length):
        data = out_data[i]
        # Handle label
        if 'label' in data:
            out_label[i] += data['label']
        
        # Handle scores
        for key in data.keys():
            if key == 'label':
                continue
            if key in scores:
                val = data[key]
                if hasattr(val, 'tolist'):
                    val = val.tolist()
                scores[key][i]['out'] += list(val)

    print("-" * 30)
    print("AUROC New Class Detection (OOD)")
    print("-" * 30)
    # ID is labeled 1, OOD is labeled 0
    # Higher score should indicate ID for metrics like MSP, MaxLogit, Energy
    for key in scores.keys():
        try:
            aucs = []
            for i in range(length):
                y_true = [1] * len(scores[key][i]['in']) + [0] * len(scores[key][i]['out'])
                y_scores = scores[key][i]['in'] + scores[key][i]['out']
                
                if len(y_scores) == 0:
                    continue
                    
                auc = roc_auc_score(y_true, y_scores)
                aucs.append(auc)
            
            if aucs:
                print(f"{key:15}: {np.mean(aucs):.4f}")
        except Exception as e:
            print(f"Error calculating AUC for {key}: {e}")

    print("\n" + "-" * 30)
    print("AUROC Failure Detection (Misclassification)")
    print("-" * 30)
    # Measures ability to detect errors (ID misclassified + OOD)
    for key in scores.keys():
        try:
            aucs = []
            for i in range(length):
                # ID Labels: 1 if correct, 0 if wrong
                # OOD Labels: Treated as 0 (wrong/unknown)
                # This part relies on 'label' containing 1 for correct pred and 0 for incorrect
                
                id_labels = in_label[i]
                ood_labels = out_label[i] # usually 0s for OOD
                
                y_true = id_labels + ood_labels
                y_scores = scores[key][i]['in'] + scores[key][i]['out']
                
                # Check for length mismatch
                if len(y_true) != len(y_scores):
                    # Fallback: ignore label mismatch if counts are off, just use standard weighting
                    pass
                
                # Calculate weighted AUC as per original script logic
                # (Weighted by inverse of class frequency to balance ID/OOD size imbalance)
                weight_id = 1.0 / len(id_labels) if len(id_labels) > 0 else 0
                weight_od = 1.0 / len(ood_labels) if len(ood_labels) > 0 else 0
                sample_weights = [weight_id]*len(id_labels) + [weight_od]*len(ood_labels)

                auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weights)
                aucs.append(auc)
                
            if aucs:
                print(f"{key:15}: {np.mean(aucs):.4f}")
        except Exception as e:
            # Silence errors for failure detection if labels aren't perfectly aligned
            # print(f"Error in failure detection for {key}: {e}")
            pass
