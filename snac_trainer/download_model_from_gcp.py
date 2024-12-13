import os
from google.cloud import storage

def download_model_from_gcp(bucket_name, output_dir='./mymodel'):
    client = storage.Client.from_service_account_json('service.json')
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='models/checkpoint-5000')
    for blob in blobs:
        local_path = os.path.join(output_dir, os.path.relpath(blob.name, 'models/checkpoint-5000'))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

# Example call:
download_folder('snac-tttts-2m-5000-1')

