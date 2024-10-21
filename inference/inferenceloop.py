import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from convert_to_wav import convert_to_wav


tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)

model_name = "amuvarma/luna-trejo-1300-vad-no-tags-3dups"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model.cuda()

prompt = '''I'm so upset that happened to you!'''
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
attention_mask = torch.ones_like(input_ids)
start_token = torch.tensor([[ 128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)

modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

input_ids = modified_input_ids
attention_mask = torch.ones_like(input_ids)

# input_ids = input_ids.to("cuda")
# attention_mask = attention_mask.to("cuda")
stop_token = 128258

start = time.time()
import torch
import torch.nn.functional as F

def custom_generate(model, input_ids, max_length=4000, temperature=0.01, top_k=50, top_p=0.95):
    device = model.device
    input_ids = input_ids.to(device)
    
    # Initialize sequence with input_ids
    generated = input_ids.clone()
    
    past_key_values = None
    eos_token_id = 128258
    target_token_id = 128257  # The token after which to print the shape
    
    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(input_ids=generated[:, -1:], use_cache=True, past_key_values=past_key_values)
            
            logits = outputs.logits[:, -1, :] / temperature
            past_key_values = outputs.past_key_values

            # Top-K filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, indices, values)
                logits = mask

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter the mask back to the original ordering
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample the next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check if the next token is the target token
            if next_token.item() == target_token_id:
                # Exclude the target token from the generated sequence
                trimmed_generated = generated[:, :-1]
                print(f"Shape after token {target_token_id}: {trimmed_generated.shape}")
                # Optionally, you can break the loop here if you want to stop generation
                # break
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == eos_token_id:
                break
    
    return generated

# Example usage:
# generated_ids = custom_generate(model, input_ids)

def remove_zeros_from_end(tensor):
    flat_tensor = tensor.flatten()
    last_nonzero = torch.where(flat_tensor != 0)[0]
    
    if len(last_nonzero) == 0:
        return torch.tensor([], device=tensor.device, dtype=tensor.dtype).reshape(1, 0)
    else:
        last_nonzero_index = last_nonzero[-1].item()
        
        result = flat_tensor[:last_nonzero_index + 1]
    return result.reshape(1, -1)

# generated_ids = custom_generate(model, input_ids)
input_ids = input_ids.cuda()
generated_ids = model.generate(input_ids, max_length=4000, pad_token_id=0, eos_token_id=128258, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, num_return_sequences=1)
print(f"time taken {time.time()-start}")
result = remove_zeros_from_end(generated_ids)
print(result.shape)

convert_to_wav(result, tokenizer)
