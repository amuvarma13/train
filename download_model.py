from huggingface_hub import hf_hub_download, HfApi
import os

base_model = "amuvarma/complete_2b-750k-interleave_x_modal-2-xyz-2"

def download_repo_files(repo_id, local_dir, exclude_files):
    api = HfApi()

    # List all files in the repository
    files = api.list_repo_files(repo_id)

    # Filter out the files to exclude
    files_to_download = [f for f in files if f not in exclude_files]

    # Download each file
    for file in files_to_download:
        try:
            print(f"Downloading: {file}")
            hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
            print(f"Downloaded: {file}")
        except Exception as e:
            print(f"Error downloading {file}: {str(e)}")

# Usage
local_dir = "./mymodel"
# exclude_files = ["checkpoint-62500/optimizer.bin", "checkpoint-62500/pytorch_model_fsdp.bin"]
exclude_files = []

# Ensure the local directory exists
os.makedirs(local_dir, exist_ok=True)

# Download files
download_repo_files(base_model, local_dir, exclude_files)

print(f"Download complete. Files saved in {local_dir}") 