import os
import timm

# 1. Set the exact paths used in your bash script
cache_dir = "/home/mauricio.alvarez/tesis/VCC/model_cache"
os.environ['TORCH_HOME'] = cache_dir
os.environ['HF_HOME'] = cache_dir

print(f"--- Cache Directory: {cache_dir} ---")

# 2. Force download of the specific model
# We set pretrained=True so it fetches from the Hub
model_name = 'vit_small_patch16_224'
print(f"Downloading {model_name}...")

try:
    # This triggers the Hugging Face Hub download
    model = timm.create_model(model_name, pretrained=True)
    print("✅ Download successful.")
except Exception as e:
    print(f"❌ Download failed: {e}")

# 3. Verify the folder was created
hf_folder = os.path.join(cache_dir, "hub", f"models--timm--{model_name}")
if os.path.exists(hf_folder):
    print(f"✅ Verified: Folder exists at {hf_folder}")
else:
    print(f"⚠️ Warning: The HF folder {hf_folder} was NOT created. Check your timm version.")