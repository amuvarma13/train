from vllm import LLM, SamplingParams
import time

def main():
    # Replace "llama-3.1-8b" with the actual Hugging Face repo ID or local model path
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Initialize the LLM
    llm = LLM(model=model_path)
    
    sampling_params = SamplingParams(
        temperature=0.7,  # Controls the "creativity" or randomness
        top_p=0.9,        # Nucleus sampling cutoff
        max_tokens=28     # Maximum number of new tokens to generate
    )

    # Prefill step - run a similar prompt to cache KV values
    # This caches computation for "Hey whats up. I think that it will actually take quite a long time to prefill the kv"
    prep = "Hey whats up. I think that it will actually take quite a long time to prefill the kv"
    outputs = llm.generate([prep], sampling_params)
    
    # Create the actual prompt (which shares tokens with the prefill prompt)
    prompt = "Hey whats up. I think that it will actually take quite a long time to prefill the kv cache."
    
    # Get tokenizer from the model to convert our text to tokens
    tokenizer = llm.get_tokenizer()
    
    # Convert the common prefix to token ids
    common_prefix = "Hey whats up. I think that it will actually take quite a long time to prefill the kv"
    prefix_token_ids = tokenizer.encode(common_prefix)
    
    # Measure performance with prefill
    start = time.monotonic()
    
    # Generate with prefill tokens - we're telling vLLM we've already computed the KV cache for these tokens
    outputs_with_prefill = llm.generate(
        prompts=[prompt],
        sampling_params=sampling_params,
        prefill_tokens=[prefix_token_ids]  # Pass the prefix token ids to reuse KV cache
    )
    
    prefill_time = time.monotonic() - start
    print("Time taken with prefill:", prefill_time)
    
    # For comparison, run without prefill
    start = time.monotonic()
    outputs_no_prefill = llm.generate([prompt], sampling_params)
    no_prefill_time = time.monotonic() - start
    print("Time taken without prefill:", no_prefill_time)
    
    # Calculate improvement
    improvement = (no_prefill_time - prefill_time) / no_prefill_time * 100
    print(f"Latency improvement: {improvement:.2f}%")
    
    # Print the responses
    print("\nPrompt:", prompt)
    print("Response with prefill:", outputs_with_prefill[0].outputs[0].text.strip())
    print("Response without prefill:", outputs_no_prefill[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()