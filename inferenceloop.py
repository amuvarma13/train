import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)

model_name = "amuvarma/luna-trejo-1300-vad-no-tags-3dups"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model.cuda()

prompt = '''That's horrible, I'm so upset that happened to you!'''
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
attention_mask = torch.ones_like(input_ids)
start_token = torch.tensor([[ 128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

input_ids = modified_input_ids
attention_mask = torch.ones_like(input_ids)

# input_ids = input_ids.to("cuda")
# attention_mask = attention_mask.to("cuda")
stop_token = 128258

start = time.time()
def custom_generate(model, input_ids, max_length=4000, temperature=0.2, top_k=50, top_p=0.95):
    device = model.device
    input_ids = input_ids.to(device)

    
    # Pre-allocate memory for the output
    output_ids = torch.zeros((1, max_length), dtype=torch.long, device=device)
    output_ids[:, :input_ids.shape[1]] = input_ids

    past_key_values = None
    for i in range(input_ids.shape[1], max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(input_ids[:, -1:], use_cache=True, past_key_values=past_key_values)

            logits = outputs.logits[:, -1, :] / temperature
            past_key_values = outputs.past_key_values

            # Apply top-k and top-p filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            top_p_mask = cumulative_probs < top_p
            top_p_mask[..., -1] = True
            filtered_logits = top_k_logits * top_p_mask.float()
            filtered_indices = top_k_indices * top_p_mask.long()

            # Sample from the filtered distribution
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            next_token = filtered_indices.gather(-1, next_token)

            output_ids[:, i] = next_token.squeeze()
            input_ids = next_token

            # if next_token.item() == tokenizer.eos_token_id:
            #     break

    return output_ids

generated_ids = custom_generate(model, input_ids)
print(f"time taken {time.time()-start}")
print(generated_ids.shape)