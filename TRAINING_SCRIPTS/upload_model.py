from huggingface_hub import HfApi
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

base_repo_id = "checkpoints/checkpoint-204000"
upload_name = "amuvarma/8b-2m-checkpoint-204000"


def upload_single_file(args):
    api, file_path, rel_path, repo_id, commit_message = args
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=rel_path,
            repo_id=repo_id,
            commit_message=commit_message
        )
        return True, rel_path
    except Exception as e:
        return False, f"Error uploading {rel_path}: {str(e)}"

def push_folder_to_hub(local_folder, repo_id, commit_message="Update model", max_workers=4):
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return None

    try:
        # Collect all files to upload
        upload_args = []
        for root, _, files in os.walk(local_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_folder)
                upload_args.append((api, file_path, rel_path, repo_id, commit_message))

        # Upload files in parallel with progress bar
        successful_uploads = 0
        failed_uploads = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(upload_args), desc="Uploading files") as pbar:
                for success, result in executor.map(upload_single_file, upload_args):
                    if success:
                        successful_uploads += 1
                    else:
                        failed_uploads.append(result)
                    pbar.update(1)

        # Print results
        print(f"\nSuccessfully uploaded {successful_uploads} files to {repo_id}")
        if failed_uploads:
            print("\nFailed uploads:")
            for error in failed_uploads:
                print(error)

        return api.get_full_repo_name(repo_id)
    except Exception as e:
        print(f"Error during upload process: {e}")
        return None

# Usage

# You can adjust max_workers based on your needs
push_folder_to_hub(
    f"./{base_repo_id}", 
    upload_name, 
    commit_message="Update model",
    max_workers=1 # Adjust this number based on your needs
)