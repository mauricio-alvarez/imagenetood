from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
import numpy as np

class ImageNet_Format(Dataset):
    """
    Wrapper for standard ImageNet Validation set (Structured: root/class/img.jpg)
    """
    def __init__(self, path, transform=None):
        self.dataset = ImageFolder(root=path, transform=transform)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class ImageNet_Subset(Dataset):
    """ Used for preprocessing (skipped, but class needed for imports) """
    def __init__(self, imagenet_path, subset_file, transform=None):
        self.root = imagenet_path
        self.transform = transform
        with open(subset_file, 'r') as f:
            self.img_list = [line.strip() for line in f]
            
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        rel_path = self.img_list[idx]
        img_path = os.path.join(self.root, rel_path)
        # Handle flat train folder if necessary
        if not os.path.exists(img_path):
             img_path = os.path.join(self.root, os.path.basename(rel_path))
             
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Dummy label
        return img, 0

class ImageNetOOD(Dataset):
    """
    Reads OOD images from a text file list.
    Handles both structured (root/class/img.jpg) and flat (root/img.jpg) folders.
    """
    def __init__(self, imagenet_path, subset_file, transform=None):
        self.root = imagenet_path
        self.transform = transform
        
        # Read the list of expected files
        with open(subset_file, 'r') as f:
            self.img_list = [line.strip() for line in f]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # The file path as listed in the text file (e.g., "n029/n029_123.JPEG")
        rel_path = self.img_list[idx]
        
        # 1. Try exact path (standard structure)
        img_path = os.path.join(self.root, rel_path)
        
        if not os.path.exists(img_path):
            # 2. Try flat path (just the filename, e.g., "n029_123.JPEG")
            filename = os.path.basename(rel_path)
            flat_path = os.path.join(self.root, filename)
            
            if os.path.exists(flat_path):
                img_path = flat_path
            else:
                # 3. Last resort: sometimes filenames have class prefixes combined or split
                # But usually option 2 covers your case.
                raise FileNotFoundError(f"Could not find image: {rel_path} or {filename} in {self.root}")

        # Load and convert
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Return 0 as dummy label for OOD
        return img, 0


class Generic_Subset(Dataset):
    def __init__(self, path, subset_file, transform=None):
        self.data = []
        self.transform = transform
        with open(subset_file) as f:
            for line in f:
                self.data.append(os.path.join(path, line.rstrip()))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
        

class ImageNetOOD_standalone(ImageFolder):
    """ If OOD is just a folder without a text list """
    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0