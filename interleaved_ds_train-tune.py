import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import ( FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
import wandb
from huggingface_hub import HfApi, create_repo
 

base_repo_id = "models"
project_name = "interleaving-datasets-tune"
resize_dataset = True

dsn1 = "amuvarma/1m-fac-raw-1dups-proc-train-col-clean"
dsn2 = "amuvarma/orcatext-dev-processed-1"
model_name = "meta-llama/Llama-3.2-3B" # Replace with your model

tokenizer_name = "meta-llama/Llama-3.2-3B"
epochs = 1
batch_size = 1
number_processes = 16
pad_token = 128263
save_steps = 12000

# torch.set_default_dtype(torch.float16)

wandb.init(project=project_name, name = "8-12-batched-alternating-r0")

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
            # print(f"returning from dataset1: {dataset_index}")
            return self.dataset1[dataset_index]
        else:
            dataset_index = super_batch * self.batch_total + (position_in_super_batch - self.batch_total)
            # print(f"returning from dataset2: {dataset_index}")
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
# model.gradient_checkpointing_enable()
# model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})



tokenizer_length = len(tokenizer)
tokens = tokenizer.convert_ids_to_tokens(range(tokenizer_length))


if resize_dataset:
    number_add_tokens = 6 * 1024 + 10

    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

ds1 = load_dataset(dsn1, split="train")

from datasets import Dataset
import numpy as np

from datasets import Dataset
import numpy as np
import os

def add_columns_to_dataset_fast(dataset):
    """
    Efficiently add labels and attention_mask columns to the dataset using batch processing
    and optimal number of CPU cores.
    """
    def add_columns_batch(examples):
        batch_size = len(examples['input_ids'])
        
        # Create labels (same as input_ids)
        labels = examples['input_ids']
        
        # Create attention_masks using list comprehension
        attention_mask = [
            [1] * len(input_ids)
            for input_ids in examples['input_ids']
        ]
        
        return {
            'input_ids': examples['input_ids'],
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    # Use all available CPU cores minus 1
    num_cpu = max(1, os.cpu_count() - 1)
    
    # Process in larger batches
    return dataset.map(
        add_columns_batch,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=num_cpu  # Dynamically set based on available cores
    )
ds1 =add_columns_to_dataset_fast(ds1)

ds2 = load_dataset(dsn2, split="train")

print(ds1, ds2)


train_dataset = BatchedAlternatingDataset(ds1, ds2, batch_total)

# dataset = dataset.shuffle(seed=42)
# train_dataset = ds1

print("Dataset loaded")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy} 


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    fp16=True,
    output_dir=f"./{base_repo_id}",
    # fsdp="full_shard",
    fsdp = "auto_wrap",

    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    # learning_rate=1e-4,

    # warmup_steps=100,
    # gradient_accumulation_steps=16,  # Adjust this value as needed
    learning_rate=3e-5,


)

print("Training arguments set")

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,  
)

# print("Trainer created")
trainer.train()
# trainer.train(resume_from_checkpoint="./mymodel")


full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = trainer.model.state_dict()

trainer.model.save_pretrained(f"./complete_{base_repo_id}", state_dict=state_dict)



