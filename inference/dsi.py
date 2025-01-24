import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Step 1: Load the tokenizer and model
mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModelForCausalLM.from_pretrained(mdn, device_map="auto", torch_dtype="auto")
print("model.dtype:", model.dtype)

# Step 2: Compile the model
compiled_model = torch.compile(model, mode='default')  # You can try different modes like 'max-autotune'

# Step 3: Prepare input
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Step 4: Measure inference time
start_time = time.time()
with torch.no_grad():
    outputs = compiled_model.generate(**inputs, max_new_tokens=500)
end_time = time.time()

# Step 5: Calculate tokens per second
output_tokens = len(outputs[0])
tokens_per_second = output_tokens / (end_time - start_time)

# Step 6: Decode and display output
print(tokenizer.decode(outputs[0]))
print(f"Tokens/second: {tokens_per_second:.2f}")
