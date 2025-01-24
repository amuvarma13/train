import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModelForCausalLM.from_pretrained(mdn)
model.eval()

# Compile the model with Torch-TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 128), dtype=torch.int64)],
    enabled_precisions={torch.float16},  # Enable FP16
)

# Prepare input
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

# Inference
with torch.no_grad():
    outputs = trt_model.generate(**inputs, max_new_tokens=500)

# Decode and display output
print(tokenizer.decode(outputs[0]))
