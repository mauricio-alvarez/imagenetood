import os
import sys

# Update these with the exact paths from your script
ID_PATH = "/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"
OOD_PATH = "/home/mauricio.alvarez/tesis/archive/imagenet-ood"

print("--- Checking ImageNet Validation Path ---")
print(f"Path: {ID_PATH}")
if os.path.exists(ID_PATH):
    contents = os.listdir(ID_PATH)
    print(f"Contains {len(contents)} items.")
    print(f"First 5 items: {contents[:5]}")
    
    # Check if it is flat or structured
    has_dirs = any(os.path.isdir(os.path.join(ID_PATH, x)) for x in contents[:10])
    has_jpgs = any(x.lower().endswith('.jpeg') or x.lower().endswith('.jpg') for x in contents[:10])
    
    if has_dirs:
        print("Status: STRUCTURED (Looks correct for PyTorch ImageFolder).")
    elif has_jpgs:
        print("Status: FLAT (Failed). This path contains images directly.")
        print("ACTION: You must organize these images into class folders (1000 folders) for the script to work.")
    else:
        print("Status: UNKNOWN/EMPTY.")
else:
    print("Status: PATH NOT FOUND.")

print("\n--- Checking ImageNet OOD Path ---")
print(f"Path: {OOD_PATH}")
if os.path.exists(OOD_PATH):
    # Check for the specific missing file from your log
    test_file = os.path.join(OOD_PATH, "n02925009/n02925009_2846.JPEG")
    print(f"Checking specific file: {test_file}")
    if os.path.exists(test_file):
        print("Status: OK. File found.")
    else:
        print("Status: MISSING FILE.")
        # Try to find where it is
        print("Searching for 'n02925009' in root...")
        if "n02925009" in os.listdir(OOD_PATH):
             print(f"Folder 'n02925009' exists in {OOD_PATH}. Checking inside...")
             print(os.listdir(os.path.join(OOD_PATH, "n02925009"))[:5])
        else:
             print(f"Folder 'n02925009' NOT found in {OOD_PATH}.")
             print(f"Contents of {OOD_PATH}[:5]: {os.listdir(OOD_PATH)[:5]}")
else:
    print("Status: PATH NOT FOUND.")
