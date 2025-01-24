from mii import pipeline
from huggingface_hub import snapshot_download

mdn = "meta-llama/Llama-3.2-3B-Instruct"
model_path = snapshot_download(
    repo_id=mdn,
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json"
    ],
    ignore_patterns=[
        "optimizer.pt",
        "pytorch_model.bin",
        "training_args.bin",
        "scheduler.pt"
    ]
)

pipe = pipeline(mdn, infer_params=True, tensor_parallel={"tp_size": 1})
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)