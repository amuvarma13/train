import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import ( FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# import wandb
from huggingface_hub import HfApi, create_repo



base_repo_id = "mymodel"

dataset_id = "amuvarma/500k-wdups-tts-1"



# model_name = "./mymodel/checkpoint-200"
model_name = "google/gemma-2-2b"
tokenizer_name = "google/gemma-2-2b"
epochs = 1
batch_size = 1
pad_token = 0
save_steps = 200





# wandb.init(
#     project=project_name, 
#     name = "run-3node-500k-post750k-fp16-eager"
#     )
 
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
model = AutoModelForCausalLM.from_pretrained(model_name)

model.gradient_checkpointing_enable() 



tokenizer_length = len(tokenizer)
model.resize_token_embeddings(tokenizer_length + number_add_tokens)

dataset = load_dataset(dataset_id, split="train")

new_dataset = dataset.select(range(0, 800))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy} 


# def preprocess_function(examples, ):
#     examples['labels'] = [
#         (token_id if token_id != pad_token else -100) for token_id in examples['input_ids']
#     ]
#     return examples



train_dataset = new_dataset
# .map(preprocess_function, batched=False, num_proc=4)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=10,
    fp16=True,

    output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    # report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True,
    # learning_rate=1e-4,
    # ignore_data_skip=True, 

    # warmup_steps=1000,
    # learning_rate=1e-6

)

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,  
)

trainer.train( resume_from_checkpoint=f"./{base_repo_id}/checkpoint-200")
# trainer.train()

# # print(trainer.model)
# num_eval_samples = 10  # You can adjust this number
# eval_dataset = dataset.shuffle(seed=42).select(range(num_eval_samples))
# eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)


# def evaluate_on_samples(model, dataset, num_samples=1):
#     model.train()  
#     device = next(model.parameters()).device 

    
    
    
    
#     total_loss = 0
    
#     with torch.no_grad():
#         for batch in tqdm(eval_dataloader, desc="Evaluating"):
#             input_ids = torch.tensor(batch['input_ids']).to(device)
#             attention_mask = torch.tensor(batch['attention_mask']).to(device)
#             labels = torch.tensor(batch['labels']).to(device)
#             if input_ids.dim() == 1:
#                 input_ids = input_ids.unsqueeze(0)
#                 attention_mask = attention_mask.unsqueeze(0)
#                 labels = labels.unsqueeze(0)
            
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()
    
#     average_loss = total_loss / num_samples
#     print(f"Average loss over {num_samples} samples: {average_loss}")

#     return average_loss


# first_eval_loss = evaluate_on_samples(trainer.model, train_dataset, num_eval_samples)
# print(f"First evaluation loss: {first_eval_loss}")


# full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

# with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
#     state_dict = trainer.model.state_dict()

# trainer.model.save_pretrained(f"./complete_{base_repo_id}", state_dict=state_dict)





# eval_loss = evaluate_on_samples(trainer.model, train_dataset, num_eval_samples)
# print(f"Evaluation complete. Average loss: {eval_loss}")


# load_repo_id = f"./complete_{base_repo_id}"
# new_model = AutoModelForCausalLM.from_pretrained(load_repo_id)
# new_model.to("cuda")

# new_eval_loss = evaluate_on_samples(new_model, train_dataset, num_eval_samples)
# print(f"New evaluation loss: {new_eval_loss}")























# # def push_folder_to_hub(local_folder, repo_id, commit_message="Update model"):
# #     api = HfApi()

# #     try:
# #         api.create_repo(repo_id=repo_id, exist_ok=True)
# #     except Exception as e:
# #         print(f"Error creating repository: {e}")
# #         return None

# #     try:
# #         uploaded_files = []
# #         for root, _, files in os.walk(local_folder):
# #             for file in files:
# #                 file_path = os.path.join(root, file)
# #                 rel_path = os.path.relpath(file_path, local_folder)
# #                 print(f"Uploading {rel_path}")
# #                 api.upload_file(
# #                     path_or_fileobj=file_path,
# #                     path_in_repo=rel_path,
# #                     repo_id=repo_id,
# #                     commit_message=commit_message
# #                 )
# #                 uploaded_files.append(rel_path)
        
# #         print(f"Successfully uploaded {len(uploaded_files)} files to {repo_id}")
# #         return api.get_full_repo_name(repo_id)
# #     except Exception as e:
# #         print(f"Error during upload: {e}")
# #         return None
    
# # push_folder_to_hub(f"./{base_repo_id}", f"amuvarma/complete_{base_repo_id}", "Update model")
