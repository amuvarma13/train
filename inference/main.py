import time


import vllm.vllm.entrypoints.llm as LLM
from vllm.vllm.entrypoints.llm import SamplingParams
from transformers import AutoTokenizer, AutoModel

mdn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModel.from_pretrained(mdn)
llm = LLM(mdn)

def generate_output(prompt, llm, sampling_params):
    start_time = time.time()
    
    # Get input token count
    input_tokens = len(tokenizer.encode(prompt))
    
    # Generate output
    output = llm.generate([prompt], sampling_params)[0]
    generated_text = output.outputs[0].text
    
    # Get output token count and calculate time
    output_tokens = len(tokenizer.encode(generated_text))
    total_tokens = input_tokens + output_tokens
    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    
    return generated_text

prompt = "Here is a short story about a dragon:"
sampling_params = SamplingParams(temperature=0.5, max_tokens=500)
generate_output(prompt, llm, sampling_params)

