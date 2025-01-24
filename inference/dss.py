import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(mdn, torch_dtype=torch.float16)

# Initialize DeepSpeed engine
engine = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=False
)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Note: engine.module is the actual sharded model
outputs = engine.module.generate(**inputs, max_new_tokens=50)
print(f"[Rank {local_rank}] {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
