from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_single_file(args):
    repo_id, filename, cache_dir = args
    try:
        # Add 'datasets/' prefix to repo_id
        full_repo_id = f"datasets/{repo_id}" if not repo_id.startswith("datasets/") else repo_id
        path = hf_hub_download(
            repo_id=full_repo_id,
            filename=filename,
            cache_dir=cache_dir,
            repo_type="dataset"  # Explicitly specify repo_type as dataset
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
    
    print(f"\nSuccessfully downloaded {len(successful_downloads)} files")
    if failed_downloads:
        print("\nFailed downloads:")
        for filename, error in failed_downloads:
            print(f"{filename}: {error}")
            
    return successful_downloads, failed_downloads

# Example usage:
repo_id = "amuvarma/proj-train-qa-and-speechqa"  # No need to add datasets/ prefix in the call
filenames = [
    "train.jsonl",
    "validation.jsonl"
]

successful, failed = download_dataset_parallel(
    repo_id=repo_id,
    filenames=filenames,
    cache_dir="./dataset_cache",
    max_workers=4
)