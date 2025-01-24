from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ORTModelForCausalLM.from_pretrained(model_name, export=True)
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="np")
outputs = model.generate(**inputs, max_new_tokens=500)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
