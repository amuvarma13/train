from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ort_model = ORTModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    optimization_level="basic"
)
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(ort_model.device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = ort_model.generate(**inputs, max_new_tokens=500)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
