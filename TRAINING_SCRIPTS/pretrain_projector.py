from datasets import load_dataset, concatenate_datasets
import wandb
import torch
import transformers
from transformers import Trainer, TrainingArguments
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import yaml
from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
)

config_file = "PRETRAIN_PROJECTOR_ARGS.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsns = config["dataset_names"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
learning_rate = config["learning_rate"]
vocab_size = config["vocab_size"]
audio_token_index = config["audio_token_index"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
audio_model_id = config["audio_model_id"]
audio_processor_id = config["audio_processor_id"]
save_folder = config["save_folder"]
batch_size = config["batch_size"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]


MODEL_FOR_CAUSAL_LM_MAPPING.register(
    "gazelle", GazelleForConditionalGeneration)


device = "cpu"
dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"Using {device} device")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print(f"Using {device} device")


config = GazelleConfig(
    audio_model_id=audio_model_id,
    text_model_id=model_name,
    audio_token_index=audio_token_index,
    vocab_size=vocab_size,

)

tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})


model = GazelleForConditionalGeneration(config).to(dtype=dtype)
model.resize_token_embeddings(len(tokenizer))
model = model.to(dtype=dtype)

for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True


wandb.init(project=project_name, name=run_name)

all_datasets = []
for dsn in dsns:
    fractional_dataset = load_dataset(dsn, split="train")
    all_datasets.append(fractional_dataset)

dataset = concatenate_datasets(all_datasets)
dataset = dataset.shuffle(seed=42)

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    audio_processor_id)


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

        if(user_response.contains("<|audio|>")):
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
    output_dir=save_folder,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,
    learning_rate=learning_rate,  # Changed to 2*10^-3
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=15000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)


trainer.train()
