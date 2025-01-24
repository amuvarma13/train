import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def measure_inference_performance(model_name, input_text):
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Warmup run
    with torch.no_grad():
        _ = model.generate(**inputs)
    
    # Timed run
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1
        )
    end_time = time.time()
    
    # Calculate metrics
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_tokens = len(outputs[0])
    inference_time = end_time - start_time
    tokens_per_second = output_tokens / inference_time
    
    return {
        'generated_text': generated_text,
        'output_tokens': output_tokens,
        'inference_time': inference_time,
        'tokens_per_second': tokens_per_second
    }

# Usage
model_name = "meta-llama/Llama-3.2-3B-Instruct"
input_text = "Input String"

results = measure_inference_performance(model_name, input_text)
print(f"Generated text: {results['generated_text']}")
print(f"Tokens/second: {results['tokens_per_second']:.2f}")
print(f"Total inference time: {results['inference_time']:.2f}s")
print(f"Output tokens: {results['output_tokens']}")