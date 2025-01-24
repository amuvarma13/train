mdn_name = "amuvarma/3b-zuckreg-convo"
from vllm import LLM, SamplingParams

def generate_output(prompt, llm, sampling_params):
   output = llm.generate([prompt], sampling_params)[0]
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   return generated_text

prompt = "The quick brown"
llm = LLM(mdn_name)
# sampling_params = SamplingParams(temperature=0.5, max_tokens=50)
# generated_text = generate_output(prompt, llm, sampling_params)
# print(generated_text)

def generate_with_embeddings(prompt, llm, model_name):
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModel.from_pretrained(model_name)
   
   inputs = tokenizer(prompt, return_tensors="pt")
   embeddings = model(**inputs).last_hidden_state
   
   sampling_params = SamplingParams(input_embeddings=embeddings)
   output = llm.generate([prompt], sampling_params)[0]
   return output.outputs[0].text

# Usage
result = generate_with_embeddings("Hello", llm, mdn_name)
print(result)