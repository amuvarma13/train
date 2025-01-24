mdn = "meta-llama/Llama-3.2-3B-Instruct"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModel.from_pretrained(mdn)
llm = LLM(mdn)
def generate_output(prompt, llm, sampling_params):
   output = llm.generate([prompt], sampling_params)[0]
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   return generated_text


