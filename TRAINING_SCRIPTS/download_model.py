from huggingface_hub import snapshot_download

# Download only model config and safetensors
model_path = snapshot_download(
    repo_id='amuvarma/zuckconvotune-3b-2m-checkpoint-124',
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ],
    ignore_patterns=[
        "optimizer.pt",
        "pytorch_model.bin",
        "training_args.bin",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.*"
    ]
)