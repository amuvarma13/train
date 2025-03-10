from vllm import LLM, SamplingParams
import time
import torch
import gc

def measure_latency(llm, prompt, sampling_params, num_runs=3, desc=""):
    """Measure average latency across multiple runs"""
    total_time = 0
    results = []
    
    for i in range(num_runs):
        # Clear CUDA cache before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        start = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        end = time.monotonic()
        
        run_time = end - start
        total_time += run_time
        results.append(outputs[0].outputs[0].text.strip())
        
        print(f"{desc} - Run {i+1}: {run_time:.4f}s")
        
        # Wait between runs
        time.sleep(0.5)
    
    avg_time = total_time / num_runs
    print(f"{desc} - Average time: {avg_time:.4f}s")
    
    return avg_time, results[-1]  # Return average time and last result

def main():
    # Replace with your model path
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    
    print("=== OPTIMIZATION TEST 1: CONTINUOUS BATCHING ===")
    # Initialize with continuous batching for lower latency
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.9,  # Higher utilization
        tensor_parallel_size=1,  # Adjust if you have multiple GPUs
        trust_remote_code=True,
        # Enable continuous batching for improved latency
        enable_chunked_prefill=True,  # Important for latency
    )
    
    # Standard sampling configuration
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128  # Generate more tokens to better measure throughput
    )
    
    # Define prompts with increasing similarity
    base_prompt = "Hey whats up. I think that it will actually take quite a long time to prefill the kv cache."
    
    # First, measure cold start latency
    print("\nMeasuring cold start latency...")
    cold_time, cold_result = measure_latency(
        llm, base_prompt, sampling_params, desc="Cold start")
    
    # Warm up with exact same prompt multiple times
    print("\nWarming up with identical prompts...")
    for i in range(3):
        _ = llm.generate([base_prompt], sampling_params)
        print(f"Warmup run {i+1} complete")
    
    # Measure latency after warmup
    print("\nMeasuring warm cache latency (same prompt)...")
    warm_time, warm_result = measure_latency(
        llm, base_prompt, sampling_params, desc="Warm cache")
    
    # Calculate improvement
    if cold_time > warm_time:
        improvement = (cold_time - warm_time) / cold_time * 100
        print(f"\nLatency improvement: {improvement:.2f}%")
    else:
        print("\nNo latency improvement detected with basic warmup")
    
    # Clean up
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)
    
    print("\n=== OPTIMIZATION TEST 2: PROMPT PACKING ===")
    # Try with an optimized configuration focusing on prompt cache
    llm = LLM(
        model=model_path,
        max_model_len=8192,  # Larger context
        gpu_memory_utilization=0.9,
        dtype="half",  # Use half precision
        trust_remote_code=True,
        # KV cache optimization parameters
        enforce_eager=True,  # Avoid CUDA graph overhead for flexible caching
        block_size=16,  # Smaller block size
        # Paged attention for more efficient memory use
        enable_prefix_caching=True,  # If available in your vLLM version
    )
    
    # Create a special sampling params for generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
        # Try with the early stopping settings
    )
    
    # Try the prompt packing approach
    print("\nUsing prompt packing approach...")
    packed_prompt = base_prompt + " " + base_prompt
    
    # Run the packed prompt first
    _ = llm.generate([packed_prompt], sampling_params)
    print("Packed prompt run complete")
    
    time.sleep(1)
    
    # Now measure with the original prompt which is contained in the packed one
    print("\nMeasuring latency after prompt packing...")
    packed_time, packed_result = measure_latency(
        llm, base_prompt, sampling_params, desc="After prompt packing")
    
    # Compare with a fresh run of a different prompt
    different_prompt = "Hello there. Can you analyze the performance characteristics of large language models?"
    print("\nMeasuring with different prompt for comparison...")
    different_time, different_result = measure_latency(
        llm, different_prompt, sampling_params, desc="Different prompt")
    
    # Calculate improvement
    if different_time > packed_time:
        improvement = (different_time - packed_time) / different_time * 100
        print(f"\nLatency improvement with packing: {improvement:.2f}%")
    else:
        print("\nNo improvement detected with prompt packing")
    
    # Print detailed stats
    print("\n=== LATENCY COMPARISON SUMMARY ===")
    print(f"Cold start latency:      {cold_time:.4f}s")
    print(f"Warm cache latency:      {warm_time:.4f}s")
    print(f"Packed prompt latency:   {packed_time:.4f}s")
    print(f"Different prompt latency: {different_time:.4f}s")
    
    # Check VRAM usage if possible
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"\nGPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")
    
    print("\nNOTE: If you're still not seeing significant improvements, your version of vLLM")
    print("might not fully support these optimizations or need additional configuration.")

if __name__ == "__main__":
    main()