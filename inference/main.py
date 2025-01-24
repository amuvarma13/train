mdn_name = "amuvarma/3b-zuckreg-convo"
from vllm import LLM, SamplingParams

def generate_output(prompt, llm, sampling_params):
   output = llm.generate([prompt], sampling_params)[0]
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   return generated_text

prompt = "The quick brown"
llm = LLM(mdn_name)
sampling_params = SamplingParams(temperature=0.5, max_tokens=50)
generated_text = generate_output(prompt, llm, sampling_params)
print(generated_text)