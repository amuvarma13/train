import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist
if dist.is_initialized():
    print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")


mdn = "meta-llama/Llama-3.2-3B-Instruct"
from mii import pipeline
pipe = pipeline(mdn)
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)