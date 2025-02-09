import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
import yaml
import wandb
from huggingface_hub import snapshot_download


config_file = "FINETUNE_ARGS-3b-speech-motion.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset"]
dsn3 = config["motion_dataset"]
resize_dataset = config["resize_dataset"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2").to(torch.bfloat16)
model.resize_token_embeddings(128266+(7*4096+10)+1000)

eval_dsn = "amuvarma/humanml3d-flat-train-padded-dedup-2"
eval_dataset = load_dataset(eval_dsn, split="train")


ds1 = load_dataset(dsn1, split="train")
ds1 = ds1.shuffle(seed=42)
ds2 = load_dataset(dsn2, split="train")
ds2 = ds2.shuffle(seed=42)
ds3 = load_dataset(dsn3, split="train")
ds3 = ds3.shuffle(seed=42)



wandb.init(project=project_name, name = run_name)

batch_total = number_processes * batch_size


class BatchedAlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, batch_total):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.batch_total = batch_total
        self.num_batches = min(
            len(dataset1) // batch_total,
            len(dataset2) // batch_total,
            len(dataset3) // batch_total
        )
        self.length = 3 * self.num_batches * batch_total

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        global_batch = index // self.batch_total
        offset = index % self.batch_total
        base_idx = (global_batch // 3) * self.batch_total
        mod = (global_batch + 1) % 3
        if mod == 1:
            return self.dataset1[base_idx + offset]
        elif mod == 2:
            return self.dataset2[base_idx + offset]
        else:
            return self.dataset3[base_idx + offset]


class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)



class MotionSpeechTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def log(self, logs, callback=None):
        super().log(logs)
        if self.is_world_process_zero():
            global_step = self.state.global_step
            mod = (global_step + 1) % 3
            if logs["loss"] is None:
                return
            if mod == 1:
                wandb.log({"motion_loss": logs["loss"], "step": global_step})
            elif mod == 2:
                wandb.log({"text_loss": logs["loss"], "step": global_step})
            else:
                wandb.log({"audio_loss": logs["loss"], "step": global_step})



tokenizer_length = len(tokenizer)
tokens = tokenizer.convert_ids_to_tokens(range(tokenizer_length))

if resize_dataset:
    number_add_tokens = 7 * 4096 + 10
    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))



train_dataset = BatchedAlternatingDataset(ds1, ds2, ds3, batch_total)
print("Dataset loaded")

def data_collator(features):
    #print keys of features
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    # Convert all lists to tensors and pad
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    # fsdp="auto_wrap",
    report_to="wandb", 
    save_steps=save_steps,
    warmup_steps=500,
    evaluation_strategy="steps",  # Evaluate every `eval_steps` during training
    eval_steps=20,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
    lr_scheduler_type="cosine"  # Cosine decay scheduler
)

print("Training arguments set")

print(eval_dataset)
trainer = MotionSpeechTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

