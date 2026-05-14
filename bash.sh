#!/bin/bash
#SBATCH --job-name=ood_vit
#SBATCH --output=logs/ood_vit_%j.log
#SBATCH --error=logs/ood_vit_%j.err
#SBATCH -p gpu                # Partition name (assuming 'gpu')
#SBATCH -c 8                  # Number of CPU cores
#SBATCH --mem=64G             # Memory request (32G is safe for a 'base' model)
#SBATCH --gres=gpu          # Request 1 GPU, erase this line if there are no standalone gpus available
#SBATCH --nodelist=g001            # Target the g002 node specifically
#SBATCH --time=10:00:00
# 1. Activate Environment

module load miniconda/3.0
conda activate imagenetood
echo "Environment is ready."

#----------------------------------------------------------------#
# Execute the Python Script
#----------------------------------------------------------------#
# Set cache variables and force offline mode to ensure timm loads from cache
export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"

# 2. Define Paths
# UPDATE THESE PATHS
IMAGENET_DIR="/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"
IMAGENET_OOD_DIR="/home/mauricio.alvarez/tesis/archive/imagenet-ood"
WEIGHTS_DIR="/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT" 
REPO_DIR=$(pwd)

# 3. Define Models
#MODELS=("vit_tiny_patch16_224|/home/mauricio.alvarez/tesis/VCC/model_cache/hub/checkpoints/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz" "vit_small_patch16_224|/home/mauricio.alvarez/tesis/VCC/model_cache/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz" "vit_base_patch16_224|/home/mauricio.alvarez/tesis/VCC/model_cache/hub/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz")
#MODELS=("vit_base_patch16_224_pruned|/home/mauricio.alvarez/tesis/sanity-check/pruned_vit_base_02.pth" "vit_small_patch16_224_pruned|/home/mauricio.alvarez/tesis/sanity-check/pruned_vit_small_02.pth")
#MODELS=("shvit_s1|/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth" "shvit_s4|/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s4.pth" "doublehead_vit|/home/mauricio.alvarez/tesis/VCC/model_weights/DoubleHeadViT/shvit_s1_doublehead_0103_300epochs_pretrained.pth")
MODELS=("doublehvit_s1|/home/mauricio.alvarez/tesis/VCC/model_weights/DoubleHeadViT/shvit_s1_doublehead_0103_300epochs_pretrained.pth" "doublehvit_s4|/home/mauricio.alvarez/tesis/VCC/3_shvit_s4_doublehead_0405_150epochs.pth" "tripleh_vit_s1|/home/mauricio.alvarez/tesis/VCC/4_shvit_s1_triplehead_0405_150epochs.pth")
# export CHECKPOINT_PATH="$WEIGHTS_DIR/$CKPT_FILE"
# 4. Loop through models
for ITEM in "${MODELS[@]}"; do
    MODEL_NAME="${ITEM%%|*}"
    CKPT_FILE="${ITEM##*|}"
    echo "========================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "Checkpoint: $CKPT_FILE"
    echo "========================================================"
    
    # Export the model name so models.py picks it up
    export MODEL_NAME=$MODEL_NAME
    export CHECKPOINT_PATH="$CKPT_FILE"
    #export IS_PRUNED="true"
    export IS_PRUNED="false"
    # Create a directory for this model's results
    RESULT_DIR="${REPO_DIR}/results/${MODEL_NAME}"
    mkdir -p "$RESULT_DIR"

   # --- Step B: Generate Scores for In-Distribution (ImageNet Val) ---
    # We treat ImageNet Validation set as the ID test set.
    # We assume 'ImageNet' dataset class handles standard ImageNet structure.
    echo "Generating ID Scores..."
    python generate_scores.py \
        --dataset ImageNet \
        --root_path "$IMAGENET_DIR" \
        --result_file "$RESULT_DIR/id_scores.pkl" \
        --semantic 0

    # --- Step C: Generate Scores for Out-Of-Distribution (ImageNet-OOD) ---
    # We use 'ImageNetOOD' dataset class. 
    # subset_file 'imagenetood.txt' is provided in the repo.
    echo "Generating OOD Scores..."
    python generate_scores.py \
        --dataset ImageNetOOD \
        --root_path "$IMAGENET_OOD_DIR" \
        --subset_file imagenetood.txt \
        --result_file "$RESULT_DIR/ood_scores.pkl" \
        --semantic 1

    # --- Step D: Evaluate ---
    # Compare ID scores vs OOD scores to get AUROC/FPR.
    echo "Evaluating..."
    python evaluate.py \
        --in_pkl "$RESULT_DIR/id_scores.pkl" \
        --out_pkl "$RESULT_DIR/ood_scores.pkl" > "$RESULT_DIR/metrics.txt"

    cat "$RESULT_DIR/metrics.txt"
    echo "Finished $MODEL"
    echo ""
done
