from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amuvarma/va-310k-320k-snac-StTtS"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64         
)

load_dataset(repo_id, split="train")
