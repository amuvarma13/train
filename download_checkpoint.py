from huggingface_hub import snapshot_download
import os

def download_model_to_local():
    # Create models directory if it doesn't exist
    local_dir = "./models"
    os.makedirs(local_dir, exist_ok=True)
    
    # Repository ID
    repo_id = "amuvarma/pretrain-360000"
    
    try:
        # Download the repository
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Set to True if you want to use symlinks
        )
        print(f"Successfully downloaded model to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    download_model_to_local()