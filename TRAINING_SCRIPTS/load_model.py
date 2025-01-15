from huggingface_hub import snapshot_download



# Download the model "google/bigbird-roberta-base" in parallel

model_path = snapshot_download("amuvarma/pretrain-snac-8b-2m-checkpoint-60000-of-299000") 
