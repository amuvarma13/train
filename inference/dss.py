import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Load tokenizer and model
mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModelForCausalLM.from_pretrained(mdn, torch_dtype=torch.float16)

# Initialize DeepSpeed Inference
ds_engine = deepspeed.init_inference(
    model,
    mp_size=2,  # Model parallel size
    dtype=torch.float16,
    replace_method='auto',
    replace_with_kernel_inject=True
)

# Prepare input
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="pt").to(ds_engine.device)

# Measure inference time
start_time = time.time()
with torch.no_grad():
    outputs = ds_engine.generate(**inputs, max_new_tokens=500)
end_time = time.time()

# Calculate tokens/second
output_tokens = len(outputs[0])
tokens_per_second = output_tokens / (end_time - start_time)

# Decode and display output
print(tokenizer.decode(outputs[0]))
print(f"Tokens/second: {tokens_per_second:.2f}")
