import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdn = "meta-llama/Llama-3.2-3B-Instruct"
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, init_inference

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModelForCausalLM.from_pretrained(mdn, device_map="auto", torch_dtype="auto")

# Initialize inference
model = init_inference(model, device_map="auto", dtype="auto")

# Inference
inputs = tokenizer("Input String", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
