import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Define your models and the specific metrics you want to analyze
models = ["shvit_s1", "shvit_s2", "shvit_s3", "shvit_s4"]
metrics = ["msp", "Maximum Cosine", "energy"] 

# Base directory for results
base_dir = "results"

# Create a big plot: Rows = Models, Cols = Metrics
fig, axes = plt.subplots(len(models), len(metrics), figsize=(18, 16))
fig.suptitle("SHViT OOD Score Distributions (ID vs OOD)", fontsize=20)

for row, model_name in enumerate(models):
    # Load Data
    id_path = os.path.join(base_dir, model_name, "id_scores.pkl")
    ood_path = os.path.join(base_dir, model_name, "ood_scores.pkl")
    
    try:
        with open(id_path, "rb") as f:
            id_data = pickle.load(f)[0]
        with open(ood_path, "rb") as f:
            ood_data = pickle.load(f)[0]
    except FileNotFoundError:
        print(f"Skipping {model_name} (files not found)")
        continue

    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        
        # Extract and Handle missing metrics (e.g. if you skipped Cosine)
        if metric in id_data:
            id_scores = np.array(id_data[metric])
            ood_scores = np.array(ood_data[metric])
            
            # Plot KDE
            sns.kdeplot(id_scores, fill=True, color="green", label="ID (ImageNet)", ax=ax, alpha=0.3)
            sns.kdeplot(ood_scores, fill=True, color="red", label="OOD", ax=ax, alpha=0.3)
            
            # Calculate overlap stats for title
            mean_diff = np.mean(id_scores) - np.mean(ood_scores)
            
            ax.set_title(f"{model_name} - {metric}\nSeparation: {mean_diff:.3f}", fontsize=10)
            
            # Only put legend on the first column to save space
            if row == 0 and col == 0:
                ax.legend(loc='upper right')
            else:
                ax.get_legend().remove() if ax.get_legend() else None
        else:
            ax.text(0.5, 0.5, "Metric N/A", ha='center')
            ax.set_title(f"{model_name} - {metric}")

        # Labeling
        if row == len(models) - 1:
            ax.set_xlabel("Score")
        if col == 0:
            ax.set_ylabel("Density")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("shvit_distribution_matrix.png")
print("Plot saved to shvit_distribution_matrix.png")