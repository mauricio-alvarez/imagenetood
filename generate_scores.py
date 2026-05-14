import sys
sys.path.append(".")
import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
# Ensure you have the models.py that supports local loading (from the previous step)
from models import get_model
from algorithms import logit_only, training_based, calculate_aurra, calculate_rmsce
from dataset import ImageNet_Format, ImageNetOOD, Generic_Subset, ImageNetOOD_standalone

def get_last_layer(model):
    if hasattr(model, 'head'):
        return model.head
    elif hasattr(model, 'fc'):
        return model.fc
    elif hasattr(model, 'classifier'):
        return model.classifier
    else:
        return list(model.children())[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)
    # preprocess_path is no longer strictly needed, but kept for compatibility
    parser.add_argument("--preprocess_path", type=str, default='.') 
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--subset_file", type=str, required=False)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--semantic", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model (Env: {os.environ.get('MODEL_NAME')})...")
    model = get_model()
    model.cuda()
    model.eval()

    # --- Setup Transforms ---
    try:
        config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**config, is_training=False)
    except:
        from torchvision.models import ResNet50_Weights
        transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

    # --- Setup Hooks ---
    temp_features = {}
    def generic_hook(module, input, output, name):
        temp_features[name] = input[0]

    last_layer = get_last_layer(model)
    last_layer.register_forward_hook(partial(generic_hook, name="feat"))

    # --- SKIPPED LOADING TRAINING STATS ---
    print("Skipping training stats loading (Running inference-only methods).")

    print(f"Loading data for {args.dataset}...")
    if args.dataset == 'ImageNet':
        dataset = ImageNet_Format(path=args.root_path, transform=transform)
    elif args.dataset == 'ImageNetOOD':
        dataset = ImageNetOOD(imagenet_path=args.root_path, subset_file=args.subset_file, transform=transform)
    elif args.dataset == 'OpenImageO':
        dataset = Generic_Subset(path=args.root_path, subset_file=args.subset_file, transform=transform)
    elif args.dataset == 'ImageNetOOD_standalone':
        dataset = ImageNetOOD_standalone(path=args.root_path, transform=transform)
    else:
        assert False, "not a valid dataset"

    # Batch size 64 is safe for ViT-Large
    out_dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4, shuffle=False
    )

    eval_features = []
    eval_logits = []
    labels = []

    print("Running Inference...")
    with torch.no_grad():
        for x, y in tqdm(out_dataloader):
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            
            eval_logits.append(pred.detach().cpu())
            eval_features.append(temp_features["feat"].detach().cpu())
            
            if args.semantic == 0:
                # ID case: Check accuracy
                labels += (torch.argmax(pred, dim=1) == y).tolist()
            else:
                # OOD case: Label placeholder
                labels += [0] * x.shape[0]

    eval_features = torch.cat(eval_features)
    eval_logits = torch.cat(eval_logits)

    # --- Compute Scores ---
    result = [{}]
    idx = 0 

    print("Calculating Scores (Inference Only)...")
    
    # 1. Maximum Cosine (Uses model weights, doesn't need train data)
    # medir el angulo entre input y imagen a clasificador
    try:
        result[idx]['Maximum Cosine'] = logit_only["Maximum Cosine"](eval_features, last_layer)
    except Exception as e:
        print(f"Skipping Max Cosine: {e}")

    # 2. ASH (Activation Shaping) - usually inference only
    #  Representacion interna al variar inputs 
    try:
        # Check if ASH is available in algorithms.training_based
        if "ash" in training_based:
            result[idx]['ash_b'] = training_based["ash"](eval_features, last_layer, 90, version='b')
    except Exception as e:
        print(f"Skipping ASH: {e}")

    # 3. MSP (Max Softmax Prob)
    result[idx]['msp'] = logit_only['Maximum Softmax Probability'](eval_logits)
    
    # 4. Max Logits
    result[idx]['ml'] = logit_only['Maximum Logits'](eval_logits)
    
    # 5. Energy Score
    result[idx]['energy'] = logit_only['Energy'](eval_logits)
    
    # REMOVED: Mahalanobis, ViM, KNN, ReAct (Require training data)

    # Store labels
    result[idx]['label'] = labels
    # --- Print Accuracy and AURRA metrics for ID/Shifted datasets ---
    if args.semantic == 0:
        try:    
            # Convert correctness boolean list to float array (1.0 and 0.0)
            correctness = np.array(labels, dtype=float)
            # MSP is standard for confidence calibration and AURRA
            msp_scores = result[idx]['msp']
            if torch.is_tensor(msp_scores):
                confidences = msp_scores.detach().cpu().numpy()
            else:
                confidences = np.array(msp_scores)
            
            acc = np.mean(correctness) * 100
            aurra = calculate_aurra(correctness, confidences) * 100
            rmsce = calculate_rmsce(correctness, confidences) * 100
            
            print("\n" + "="*30)
            print(f"{args.dataset} Results")
            print(f"Accuracy (%):\t\t {acc:.2f}")
            print(f"RMS Calib Error (%):\t {rmsce:.2f}")
            print(f"AURRA (%):\t\t {aurra:.2f}")
            print("="*30 + "\n")

        except Exception as e:
            print(f"\n[ERROR] Failed to calculate metrics: {e}\n")
    # Save
    print(f"Saving scores to {args.result_file}...")
    with open(args.result_file, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
