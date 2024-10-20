import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)

model_name = "amuvarma/luna-trejo-1300-vad-no-tags-3dups"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.cuda()

prompt = '''That's horrible, I'm so upset that happened to you!'''
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
attention_mask = torch.ones_like(input_ids)
start_token = torch.tensor([[ 128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

input_ids = modified_input_ids
attention_mask = torch.ones_like(input_ids)

input_ids = input_ids.to("cuda")
attention_mask = attention_mask.to("cuda")
stop_token = 128258

start = time.time()
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=4000,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.05,
    eos_token_id=stop_token,
    # bad_words_ids=[[129162, 128911]]
)

print(f"time taken {time.time()-start}")
print(generated_ids.shape)