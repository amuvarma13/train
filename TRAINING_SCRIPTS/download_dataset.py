from huggingface_hub import snapshot_download

repo_id = "amuvarma/mls-eng-10k-500k-projection_prep"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64         
)
