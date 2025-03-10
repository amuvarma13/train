from vllm import LLM, SamplingParams
import time

def main():
    # Replace "llama-3.1-8b" with the actual Hugging Face repo ID or local model path
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Initialize the LLM
    llm = LLM(model=model_path)
    
    sampling_params = SamplingParams(
        temperature=0.7,  # Controls the “creativity” or randomness
        top_p=0.9,        # Nucleus sampling cutoff
        max_tokens=28     # Maximum number of new tokens to generate
    )

    prep = "Hey whats up. I think that it will actually take quite a long time to prefill the kv"
    outputs = llm.generate([prep], sampling_params)
    
    # Create a prompt
    start = time.monotonic()
    prompt = "Hey whats up. I think that it will actually take quite a long time to prefill the kv cache."
    
    # Configure the sampling parameters

    
    # Generate text
    outputs = llm.generate([prompt], sampling_params)
    print("Time taken:", time.monotonic() - start)
    
    # Print the first (and only) response
    print("Prompt:", prompt)
    print("Response:", outputs[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()
