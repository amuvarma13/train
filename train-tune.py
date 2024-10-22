import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from huggingface_hub import HfApi, create_repo

base_repo_id = "models"
project_name = "luna-tune"
dsn = "amuvarma/luna-2.6k-tts-1-wtags-vad-weval"

model_name = "amuvarma/llama-2.3m-full"
tokenizer_name = "meta-llama/Llama-3.2-3B"
epochs = 1
batch_size = 4
pad_token = 128263
save_steps = 3000
validation_split = 0.05

wandb.init(
    project=project_name,
    name="run-3b-2.6k-"
)

number_add_tokens = 6 * 1024 + 10

class FSDPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = base_repo_id
        self.api = HfApi()

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.save_and_push_model(output_dir)

    def save_and_push_model(self, output_dir):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)

# Load and tokenize
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
model.gradient_checkpointing_enable()

tokenizer_length = len(tokenizer)
tokens = tokenizer.convert_ids_to_tokens(range(tokenizer_length))

new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Load and split dataset
dataset = load_dataset(dsn, split="train")
dataset = dataset.shuffle(seed=42)
split_size = int(len(dataset) * (1 - validation_split))
train_dataset = dataset.select(range(split_size))
eval_dataset = dataset.select(range(split_size, len(dataset)))

print(f"Train size: {len(train_dataset)}, Validation size: {len(eval_dataset)}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Training arguments without evaluation
training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=12,
    fp16=True,
    output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
)

print("Training arguments set")

# Create trainer without eval_dataset
trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

# Perform training
trainer.train()

print("Training completed. Starting validation...")

# Create separate evaluation args
eval_args = TrainingArguments(
    output_dir="./eval_output",
    per_device_eval_batch_size=1,  # Small batch size for evaluation
    remove_unused_columns=True,
    fp16=True,
    fsdp="auto_wrap"
)

# Create separate evaluation trainer
eval_trainer = FSDPTrainer(
    model=model,
    args=eval_args,
    compute_metrics=compute_metrics,
)

# Run evaluation
eval_results = eval_trainer.evaluate(eval_dataset=eval_dataset)
print("Validation Results:", eval_results)
wandb.log({"final_evaluation": eval_results})

# Save the final model
full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = trainer.model.state_dict()

trainer.model.save_pretrained(f"./complete_{base_repo_id}", state_dict=state_dict)