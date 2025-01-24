from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ort_model = ORTModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    providers=["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
)
input_text = "Here is a short story about a dragon:"
inputs = tokenizer(input_text, return_tensors="np")
outputs = ort_model.generate(**inputs, max_new_tokens=500, use_cache=False)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
