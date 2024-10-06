import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import ( FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from huggingface_hub import HfApi, create_repo



base_repo_id = "2.3m-test-0"
project_name = "3dups"
dataset_id = "amuvarma/2.3m-llama-wdups"

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct
epochs = 1
batch_size = 1
pad_token = 0
save_steps = 4000


wandb.init(
    project=project_name, 
    name = "2nodeswdupsrunpodllama3bflashattn"
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





tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
# model.gradient_checkpointing_enable()



tokenizer_length = len(tokenizer)
tokens = tokenizer.convert_ids_to_tokens(range(tokenizer_length))


new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset(dataset_id, split="train")

print("Dataset loaded")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy} 

train_dataset = dataset
training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=60,
    fp16=True,
    output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    # learning_rate=7e-5,


)

print("Training arguments set")

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,  
)

print("Trainer created")
trainer.train()
# trainer.train(resume_from_checkpoint="./2.3m-test-0/checkpoint-4000")


full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = trainer.model.state_dict()

trainer.model.save_pretrained(f"./complete_{base_repo_id}", state_dict=state_dict)


def push_folder_to_hub(local_folder, repo_id, commit_message="Update model"):
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return None

    try:
        uploaded_files = []
        for root, _, files in os.walk(local_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_folder)
                print(f"Uploading {rel_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    commit_message=commit_message
                )
                uploaded_files.append(rel_path)
        
        print(f"Successfully uploaded {len(uploaded_files)} files to {repo_id}")
        return api.get_full_repo_name(repo_id)
    except Exception as e:
        print(f"Error during upload: {e}")
        return None
    
push_folder_to_hub(f"./{base_repo_id}", f"amuvarma/complete_{base_repo_id}", "Update model")
