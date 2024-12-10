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
import wandb
from huggingface_hub import HfApi, create_repo

base_repo_id = "models"
project_name = "interleaving-datasets-pretrain"
resize_dataset = True

dsn1_1 = "amuvarma/CanopyLabs-audio_pretrain_10m-facodec-1dups-proc"
dsn1_2 = "amuvarma/CanopyElias-audio_pretrain_10m-facodec-1dups-proc"
dsn1_3 = "amuvarma/eliasfiz-audio_pretrain_10m-facodec-1dups-proc"

ds1_1 = load_dataset(dsn1_1, split="train")
ds1_2 = load_dataset(dsn1_2, split="train")
ds1_3 = load_dataset(dsn1_3, split="train")

dsn2_0 = "amuvarma/text-messages-6m-processed-1"
ds2_0 = load_dataset(dsn2_0, split="train")
ds2_1 = ds2_0.shuffle(seed=42)
ds2 = concatenate_datasets([ds2_0, ds2_1])

model_name = "meta-llama/Llama-3.2-3B" # Replace with your model

tokenizer_name = "meta-llama/Llama-3.2-3B"
epochs = 1
batch_size = 1
number_processes = 24
pad_token = 128263
save_steps = 12000

wandb.init(project=project_name, name = "9-12-batched-alternating-r2")

batch_total = number_processes * batch_size

class BatchedAlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.length = 2 * min(len(dataset1), len(dataset2))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        super_batch = index // (2 * self.batch_total)
        position_in_super_batch = index % (2 * self.batch_total)
        
        if position_in_super_batch < self.batch_total:
            dataset_index = super_batch * self.batch_total + position_in_super_batch
            return self.dataset1[dataset_index]
        else:
            dataset_index = super_batch * self.batch_total + (position_in_super_batch - self.batch_total)
            return self.dataset2[dataset_index]

class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class FSDPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = base_repo_id
        self.api = HfApi()
    
    def get_train_dataloader(self):
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False, 
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if self.is_world_process_zero():
            global_step = self.state.global_step
            if global_step % 2 == 0:
                wandb.log({"text_loss": logs["loss"], "step": global_step})
            else:
                wandb.log({"audio_loss": logs["loss"], "step": global_step})

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.save_and_push_model(output_dir)
 
    def save_and_push_model(self, output_dir):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

tokenizer_length = len(tokenizer)
tokens = tokenizer.convert_ids_to_tokens(range(tokenizer_length))

if resize_dataset:
    number_add_tokens = 6 * 1024 + 10
    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

ds1 = concatenate_datasets([ds1_1, ds1_2, ds1_3])
ds1 = ds1.shuffle(seed=42)

# Remove the preprocessing step:
# ds1 = add_columns_to_dataset_fast(ds1)

print(ds1, ds2)

train_dataset = BatchedAlternatingDataset(ds1, ds2, batch_total)
print("Dataset loaded")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy} 

# Define a simple data collator
def data_collator(features):
    # Ensure we have input_ids
    input_ids = [f["input_ids"] for f in features]

    # Create attention_mask if not present
    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    # Create labels if not present
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
    fp16=True,
    output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    # learning_rate=1e-4,
    learning_rate=8e-5,  # 8e-4 is the learning rate used in the original Llama paper
    warmup_ratio=0.03,  # 3% of total steps
    lr_scheduler_type="cosine"  # Cosine decay scheduler
)

print("Training arguments set")

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,  
    data_collator=data_collator,  # <-- Add this line
)

trainer.train()

full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = trainer.model.state_dict()
trainer.model.save_pretrained(f"./complete_{base_repo_id}", state_dict=state_dict)
