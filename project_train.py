from datasets import load_dataset, concatenate_datasets
import wandb
import torch
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import yaml
from orpheus import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
import whisper

whisper_model = whisper.load_model("small")
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<|audio|>"]}
)

vs = 152064

print("tokeniser is length of", len(tokenizer))

config = OrpheusConfig(
            text_model_id=model_name,
            audio_token_index=152064,
            vocab_size=152064,
            hidden_size=3584,
            audio_hidden_size=1024,
        )

model = OrpheusForConditionalGeneration(config)
print(model)


dsn = "amuvarma/mls-eng-10k-500k-projection_prep"
ds = load_dataset(dsn, split="train")

for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True

wandb.init(project="test-proj-8", name="r1")

class AudioChatDataCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.whisper_model = whisper_model.to("cuda")
        self.model = model
        
        pass

    def _process_audio_tensor(self, audio, sample_rate=16000):
        audio = audio.to(torch.float32)
        duration_ms = (len(audio) / sample_rate) * 1000
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel, int(duration_ms / 20) + 1

    def _inference_collator(self, audio_input, user_res, ass_res):
        user_input_ids = self.tokenizer(
            user_res, return_tensors="pt").input_ids
        assistant_input_ids = self.tokenizer(
            ass_res, return_tensors="pt").input_ids

        # start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor(
            [[128009]], dtype=torch.int64)
        # final_tokens = torch.tensor([[128009]], dtype=torch.int64)

        user_tokens = torch.cat(
            [user_input_ids, end_tokens], dim=1)

        labels = torch.cat([user_input_ids, end_tokens,
                            assistant_input_ids], dim=1)

        true_labels = torch.full_like(labels, -100)
        true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

        attention_mask = torch.ones_like(labels)

        audio_input = audio_input.squeeze(0)
        mel, length = self._process_audio_tensor(audio_input)
        mel = mel.to(whisper_model.device)
        mel = mel.unsqueeze(0)
        with torch.no_grad():
            audio_feature = whisper_model.embed_audio(mel)[0][:length]
        audio_feature = audio_feature.unsqueeze(0)


        return {
            "audio_values": audio_feature.to(self.model.device).to(self.model.dtype),
            "input_ids": labels.to(self.model.device),
            "labels": true_labels.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device)
        }

    def __call__(self, features):
        audio = torch.tensor([features[0]["audio"]["array"]])
        assistant_response = features[0]["assistant"]
        user_response = features[0]["user"]

        # Simple contains check
        if "<|audio|>" in user_response:
            user_response = features[0]["user"]
        else:
            user_response = "<|audio|>"

        batch = self._inference_collator(
            audio, user_response, assistant_response)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }



training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    warmup_ratio=0.03,
    learning_rate=2e-06,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=15000
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=AudioChatDataCollator(tokenizer, model)
)


trainer.train()