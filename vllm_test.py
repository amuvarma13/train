from vllm import LLM, SamplingParams
import time

def main():
    # Replace with the actual Hugging Face repo ID or local model path
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Initialize the LLM with cache optimization parameters
    llm = LLM(
        model=model_path,
        # These parameters can help with KV cache efficiency
        max_model_len=4096,  # Adjust based on your needs
        gpu_memory_utilization=0.8,  # Use more GPU memory for caching
        # Uncomment if these parameters are available in your vLLM version
        # enforce_eager=True,  # May help with cache performance
        # block_size=16,  # Smaller block size for more efficient memory usage
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=28
    )

    # First run - execute a similar prefix to populate the KV cache
    # This exact prompt will warm up the cache with most of the tokens
    prep = "Hey whats up. I think that it will actually take quite a long time to prefill the kv"
    print("Warming up KV cache...")
    warmup_start = time.monotonic()
    outputs_warmup = llm.generate([prep], sampling_params)
    warmup_time = time.monotonic() - warmup_start
    print(f"Warmup time: {warmup_time:.4f} seconds")
    
    # Wait briefly to ensure all operations are complete
    time.sleep(0.5)
    
    # Create the actual prompt (which shares tokens with the prefill prompt)
    prompt = "Hey whats up. I think that it will actually take quite a long time to prefill the kv cache."
    
    # Run the full query - should benefit from cached KV computations
    print("\nRunning with warmed-up cache...")
    start = time.monotonic()
    outputs_with_cache = llm.generate([prompt], sampling_params)
    cached_time = time.monotonic() - start
    print(f"Time with warmed cache: {cached_time:.4f} seconds")
    
    # Compare with a cold run with a different prompt that doesn't share tokens
    # but is similar in length to ensure fair comparison
    cold_prompt = "Hello there. I believe the process of computing attention values requires significant time."
    
    # Clear any existing KV caches, if the API supports it
    # If not available, instantiate a new model
    del llm
    time.sleep(1)  # Give time for cleanup
    
    # Create a new model instance to ensure cold start
    llm = LLM(model=model_path)
    
    print("\nRunning with cold cache...")
    cold_start = time.monotonic()
    outputs_cold = llm.generate([cold_prompt], sampling_params)
    cold_time = time.monotonic() - cold_start
    print(f"Time with cold cache: {cold_time:.4f} seconds")
    
    # Calculate approximate improvement
    if cold_time > cached_time:
        improvement = (cold_time - cached_time) / cold_time * 100
        print(f"\nApproximate latency improvement: {improvement:.2f}%")
    else:
        print("\nNo measurable improvement detected")
    
    # Print the responses
    print("\nWarm-up prompt:", prep)
    print("Warm-up response:", outputs_warmup[0].outputs[0].text.strip())
    
    print("\nActual prompt:", prompt)
    print("Response with warmed cache:", outputs_with_cache[0].outputs[0].text.strip())
    
    print("\nCold prompt:", cold_prompt)
    print("Response with cold cache:", outputs_cold[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()