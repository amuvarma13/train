import os
from google.cloud import storage
from tqdm import tqdm

def upload_folder(bucket_name, folder_path, prefix=''):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            all_files.append(os.path.join(root, f))

    client = storage.Client.from_service_account_json('service.json')
    bucket = client.bucket(bucket_name)

    for local_path in tqdm(all_files, desc="Uploading files"):
        remote_path = os.path.join(prefix, os.path.relpath(local_path, folder_path)).replace('\\', '/')
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)



upload_folder('snac-tttts-2m-5000-1', './models/checkpoint-5000', 'models/checkpoint-5000')