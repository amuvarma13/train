import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(mdn, torch_dtype=torch.float16)

# Initialize DeepSpeed inference engine
engine = deepspeed.init_inference(
    model,
    mp_size=2,               # 2 GPUs
    dtype=torch.float16,     # or bfloat16
    replace_method="auto",
    replace_with_kernel_inject=True
)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(engine.device)
outputs = engine.module.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
