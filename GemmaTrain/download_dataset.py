from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amuvarma/all-texts-2048-iids"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
) 

ds = load_dataset(repo_id, split="train")
print(ds)
 