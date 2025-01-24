import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(mdn, torch_dtype=torch.bfloat16)

model = torch.compile(model)
# Initialize DeepSpeed engine
engine = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=False
)

prompt = "Here is a story about a dragon:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Note: engine.module is the actual sharded model
start_time = time.time()
with torch.inference_mode():
    outputs = engine.module.generate(**inputs, max_new_tokens=500)
end_time = time.time()
print("Tokens/second:", len(outputs[0]) / (end_time - start_time))
print(f"[Rank {local_rank}] {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
