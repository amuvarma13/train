from huggingface_hub import snapshot_download

# Replace "username/dataset_name" with the actual repo ID
repo_id = "gpt-omni/VoiceAssistant-400K" # e.g., "huggingface/poems_dataset"

# Download the entire dataset repository
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",    # important if it's a dataset repo
    revision="main",        # or a specific branch/tag/commit
    max_workers=64          # number of parallel download threads
)
