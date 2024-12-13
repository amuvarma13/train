import os
from google.cloud import storage

def upload_folder(bucket_name, folder_path, prefix=''):
    client = storage.Client.from_service_account_json('service.json')
    bucket = client.bucket(bucket_name)
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            local_path = os.path.join(root, f)
            remote_path = os.path.join(prefix, os.path.relpath(local_path, folder_path)).replace('\\', '/')
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)


upload_folder('snac-tttts-2m-5000', './models/checkpoint-5000', 'models/checkpoint-5000')