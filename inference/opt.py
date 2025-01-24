from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Make sure to disable the fast tokenizer if you run into tokenizer-related issues,
# especially for LLaMA-based models
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Export the model to ONNX
ort_model = ORTModelForCausalLM.from_pretrained(model_name, export=True)

# Disable use of cache in the config
ort_model.config.use_cache = False

input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="np")

# Also explicitly pass `use_cache=False` here:
outputs = ort_model.generate(**inputs, max_new_tokens=100, use_cache=False)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
