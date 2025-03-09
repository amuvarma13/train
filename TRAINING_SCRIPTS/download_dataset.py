from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amuvarma/emilia-snac-merged-18m-mod7-delay-TTS-all-rows-8192"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=10,     
) 

load_dataset(repo_id, split="train")
 