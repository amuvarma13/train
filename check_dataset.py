from datasets import load_dataset
repo_id = "amuvarma/proj-train-qa-and-speechqa"  # e.g., "huggingface/poems_dataset"
dataset = load_dataset(repo_id, split="train")