import pickle
import numpy as np
import os

# Define your paths
base_dir = "results" # or wherever your results folder is
models = [
    "vit_tiny_patch16_224", 
    "vit_small_patch16_224", 
    "vit_base_patch16_224", 
    "vit_large_patch16_224",
    "shvit_s1",
    "shvit_s2",
    "shvit_s3",
    "shvit_s4",
    "doublehead_vit"
]

print(f"{'Model':<30} | {'Top-1 Accuracy':<15}")
print("-" * 50)

for model in models:
    file_path = os.path.join(base_dir, model, "id_scores.pkl")
    
    if not os.path.exists(file_path):
        print(f"{model:<30} | Not Found")
        continue
        
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        # The 'label' list contains 1 for correct prediction, 0 for incorrect
        # So the mean of this list is the accuracy.
        acc = np.mean(data[0]['label']) * 100
        
    print(f"{model:<30} | {acc:.2f}%")