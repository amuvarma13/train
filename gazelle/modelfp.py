# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

model_pms = 3

model_name = f"meta-llama/Llama-3.2-{model_pms}B"
if(model_pms == 3):
    hidden_size = 3072
elif model_pms == 8:
    hidden_size = 4096
else:
    hidden_size = 2048

print("the hidden size is", hidden_size)    



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
w2vmodel = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")



class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class GazelleLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = model
        self.multimodal_projector = nn.Sequential(
            RMSNorm(),
            nn.Linear(6144, hidden_size, bias=False),
            SwiGLU(),
            nn.Linear(hidden_size//2, hidden_size, bias=False),
            RMSNorm()
        )
        
        self.stack_factor = 8
        self.audio_tower = w2vmodel
        self.pad_length = 1000

        for param in self.llm.parameters():
            param.requires_grad = False

        for param in self.audio_tower.parameters():
            param.requires_grad = False
        

    def _pad_and_stack(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        audio_embeds = F.pad(
            audio_embeds, (0, 0, 0, self.stack_factor - T % self.stack_factor)
        )
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
        return audio_embeds

    def _pad_and_crop(self, combined_features, labels=None, transcript_length=None):
        b, n, d = combined_features.shape
        
        # First handle the features padding/cropping
        if n > self.pad_length:
            combined_features = combined_features[:, :self.pad_length, :]
            attention_mask = torch.ones(b, self.pad_length, device=combined_features.device)
        elif n < self.pad_length:
            pad_size = self.pad_length - n
            padding = torch.zeros(b, pad_size, d, device=combined_features.device, dtype=combined_features.dtype)
            combined_features = torch.cat([combined_features, padding], dim=1)
            attention_mask = torch.cat([
                torch.ones(b, n, device=combined_features.device),
                torch.zeros(b, pad_size, device=combined_features.device)
            ], dim=1)
        else:
            attention_mask = torch.ones(b, self.pad_length, device=combined_features.device)

        if labels is not None and transcript_length is not None:
            full_labels = torch.full((b, self.pad_length), -100, 
                                  device=combined_features.device, 
                                  dtype=labels.dtype)
            
            transcript_start = n - transcript_length
            
            if transcript_start < self.pad_length:
                end_idx = min(transcript_start + transcript_length, self.pad_length)
                full_labels[:, transcript_start:end_idx] = labels[:, :end_idx-transcript_start]
                
            return combined_features, attention_mask, full_labels
            
        return combined_features, attention_mask

      
    def forward(
        self,
        input_ids=None,
        transcript_ids = None,
        audio_values = None,
        attention_mask=None,
        labels=None,
    ):
        clean_transcript_ids = (transcript_ids[transcript_ids != 128001]).unsqueeze(0)
        # print("shapes", clean_transcript_ids.shape, transcript_ids.shape)

        resp_toks = torch.tensor([[128000, 578, 17571, 358, 1120, 1071, 574, 25, 220]])
        resp_toks = resp_toks.to(audio_values.device)



        input_embeds = self.llm.model.embed_tokens(input_ids)
        transcript_embeds = self.llm.model.embed_tokens(clean_transcript_ids)

        resp_embeds = self.llm.model.embed_tokens(resp_toks)    

        audio_embeds = self.audio_tower(audio_values)
        audio_embeds_lhs = audio_embeds.last_hidden_state
        audio_embs_reshaped = self._pad_and_stack(audio_embeds_lhs)
        audio_features = self.multimodal_projector(audio_embs_reshaped)

        
        combined_features = torch.cat([audio_features, input_embeds, resp_embeds, transcript_embeds], dim=1)
        attention_mask = torch.ones_like(combined_features[:, :, 0])

        output = self.llm(
            input_ids=None,
            inputs_embeds=combined_features,
            attention_mask=attention_mask
        )

        #now process output for sft loss calculation
        
        loss_rel_outputs  = output.logits[:, -clean_transcript_ids.shape[1]:, :]

        # print("output", output.logits.shape, loss_rel_outputs.shape, clean_transcript_ids.shape)



        if labels is not None:
              shift_logits = loss_rel_outputs[..., :-1, :].contiguous()
              shift_labels = clean_transcript_ids[..., 1:].contiguous()
              
              shift_logits = shift_logits.view(-1, shift_logits.size(-1))
              shift_labels = shift_labels.view(-1)
              
              loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
              loss = loss_fct(shift_logits, shift_labels)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )