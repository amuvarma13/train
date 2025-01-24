from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModelForCausalLM.from_pretrained(mdn)

# Export to ONNX with Optimum
ort_model = ORTModelForCausalLM._from_transformers(model, export=True, save_dir="./onnx_model")

# Inference with ONNX Runtime
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="np")

outputs = ort_model.generate(**inputs, max_new_tokens=500)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
