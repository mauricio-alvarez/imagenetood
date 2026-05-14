import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Settings
model = "vit_tiny_patch16_224"  # Change to the model you want to check
score_type = "Maximum Cosine"    # This was your best performing metric
results_dir = f"results/{model}"

# Load Data
with open(os.path.join(results_dir, "id_scores.pkl"), "rb") as f:
    id_data = pickle.load(f)[0]
with open(os.path.join(results_dir, "ood_scores.pkl"), "rb") as f:
    ood_data = pickle.load(f)[0]

# Extract scores
id_scores = np.array(id_data[score_type])
ood_scores = np.array(ood_data[score_type])

# Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(id_scores, fill=True, color="green", label="In-Distribution (ImageNet)")
sns.kdeplot(ood_scores, fill=True, color="red", label="Out-of-Distribution (OOD)")

plt.title(f"{score_type} Distribution - {model}")
plt.xlabel("Score (Higher = More Confident)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)

# Save
output_file = f"plot_{model}_{score_type.replace(' ', '_')}.png"
plt.savefig(output_file)
print(f"Plot saved to {output_file}")