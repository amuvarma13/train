from huggingface_hub import snapshot_download

repo_id = "amuvarma/proj-train-qa-and-speechqa"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64         
)
