from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amuvarma/em-EN-TTS-full-8192"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
) 

load_dataset(repo_id, split="train")
 