from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_single_file(args):
    repo_id, filename, cache_dir = args
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        return True, filename, path
    except Exception as e:
        return False, filename, str(e)

def download_dataset_parallel(repo_id, filenames, cache_dir=None, max_workers=4):
    # Prepare arguments for parallel download
    download_args = [
        (repo_id, filename, cache_dir) 
        for filename in filenames
    ]
    
    successful_downloads = []
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(filenames), desc="Downloading files") as pbar:
            for success, filename, result in executor.map(download_single_file, download_args):
                if success:
                    successful_downloads.append((filename, result))
                else:
                    failed_downloads.append((filename, result))
                pbar.update(1)
    
    # Print results
    print(f"\nSuccessfully downloaded {len(successful_downloads)} files")
    if failed_downloads:
        print("\nFailed downloads:")
        for filename, error in failed_downloads:
            print(f"{filename}: {error}")
            
    return successful_downloads, failed_downloads

# Example usage:
repo_id = "amuvarma/proj-train-qa-and-speechqa"
filenames = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "metadata.json"
]

successful, failed = download_dataset_parallel(
    repo_id=repo_id,
    filenames=filenames,
    max_workers=4
)