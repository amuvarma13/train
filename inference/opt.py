from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
ort_model = ORTModelForCausalLM.from_pretrained(
    model,
    export=True,
    optimization_level="basic",
    providers=providers
)
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = ort_model.generate(**inputs, max_new_tokens=500)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
