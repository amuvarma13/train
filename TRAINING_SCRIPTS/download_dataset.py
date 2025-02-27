from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "gpt-omni/VoiceAssistant-400K"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

load_dataset(repo_id, split="train")
