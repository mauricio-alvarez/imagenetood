import sys
sys.path.append(".")
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Import the get_model function we defined in step 1
# If you haven't updated models.py yet, ensure it has the get_model function provided in the guide.
from models import get_model 
from dataset import ImageNet_Subset

def get_last_layer(model):
    """Helper to get the final linear layer for hooking."""
    if hasattr(model, 'head'):
        return model.head
    elif hasattr(model, 'fc'):
        return model.fc
    elif hasattr(model, 'classifier'):
        return model.classifier
    else:
        # Fallback for some timm models or older torchvision
        return list(model.children())[-1]

if __name__ == "__main__":
    print("started preprocess")
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)
    parser.add_argument("--imagenet_path", type=str, required=True)
    parser.add_argument("--subset_file", type=str, required=True)
    parser.add_argument("--result_path", type=str, default='.')
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)

    print("loading model...")
    # Load the single model specified by env var in the run script
    model = get_model() 
    model.cuda()
    model.eval()

    # --- Setup Transforms specific to the model ---
    # This ensures ViT gets 224x224 and correct normalization
    try:
        config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**config, is_training=False)
    except:
        # Fallback if config isn't available (e.g. old torchvision)
        from torchvision.models import ResNet50_Weights
        transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
    
    print(f"Using transform: {transform}")

    # --- Setup Feature Hooks ---
    temp_features = {}
    def generic_hook(module, input, output, name):
        # input is a tuple, we want the tensor
        temp_features[name] = input[0]

    last_layer = get_last_layer(model)
    print(f"Hooking into layer: {last_layer}")
    last_layer.register_forward_hook(partial(generic_hook, name="feat"))

    print("loading data...")
    train_dataset = ImageNet_Subset(
        imagenet_path=args.imagenet_path, 
        subset_file=args.subset_file, 
        transform=transform
    )
    # Reduced batch size slightly to be safe with ViT-Large
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, num_workers=4, shuffle=False
    )

    train_features = []
    train_logits = []
    train_y = []

    print("Extracting features...")
    with torch.no_grad():
        for x, y in tqdm(train_dataloader):
            x = x.cuda()
            
            # Forward pass
            logits = model(x)
            
            # Store results
            train_logits.append(logits.detach().cpu())
            train_features.append(temp_features["feat"].detach().cpu())
            train_y.extend(y.cpu().detach().numpy().tolist())

    # Concatenate
    train_features = torch.cat(train_features)
    train_logits = torch.cat(train_logits).numpy()
    train_y = np.array(train_y)

    print(f"Saving to {args.result_path}...")
    # We save as index '0' or without index. 
    # To keep simple, we assume the run script handles folder separation,
    # so we just name them consistently.
    torch.save(train_logits, os.path.join(args.result_path, "train_logits.pt"))
    torch.save(train_features, os.path.join(args.result_path, "train_features.pt"))
    np.save(os.path.join(args.result_path, "train_y.npy"), train_y)
    
    print("Preprocess finished.")
