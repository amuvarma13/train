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
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "amuvarma/3b-10m-pretrain-full"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = OrpheusConfig(
            text_model_id=model_name,
            audio_token_index=156939,
            vocab_size=156939,
            hidden_size=3072,
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

wandb.init(project="test-proj", name="r0")


def inference_collator(audio_input, user_res, ass_res):

    user_input_ids = tokenizer(user_res, return_tensors="pt").input_ids
    assistant_input_ids = tokenizer(ass_res, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009]], dtype=torch.int64)

    user_tokens = torch.cat(
        [start_token, user_input_ids, end_tokens], dim=1)

    labels = torch.cat([start_token, user_input_ids, end_tokens,
                       assistant_input_ids, final_tokens], dim=1)

    true_labels = torch.full_like(labels, -100)
    true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

    attention_mask = torch.ones_like(labels)


    return {
        "audio_values": audio_input.to(model.device).to(model.dtype),
        "input_ids": labels.to(model.device),
        "labels": true_labels.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }



class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):
        audio = torch.tensor([features[0]["audio"]["array"]])
        assistant_response = features[0]["assistant"]
        user_response = features[0]["user"]

        # Simple contains check
        if "<|audio|>" in user_response:
            user_response = features[0]["user"]
        else:
            user_response = "<|audio|>"
            
    
        

        batch = inference_collator(audio, user_response, assistant_response)

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
    lr_scheduler_type="cosine",
    fp16=True,
    save_steps=15000
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=AudioChatDataCollator(),
)


trainer.train()