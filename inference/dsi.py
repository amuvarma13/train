import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModelForCausalLM.from_pretrained(mdn, device_map="auto", torch_dtype="auto")
#print dtype
print("model.dtype:", model.dtype)

# Initialize inference
inputs = tokenizer("Here is a short story about a dragon:", return_tensors="pt").to(model.device)

# Measure time
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=500)
end_time = time.time()

# Calculate tokens/second
output_tokens = len(outputs[0])
tokens_per_second = output_tokens / (end_time - start_time)

print(tokenizer.decode(outputs[0]))
print(f"Tokens/second: {tokens_per_second:.2f}")