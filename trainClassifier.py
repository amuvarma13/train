import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import wandb
from huggingface_hub import HfApi, create_repo
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import ( FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
base_repo_id = "models"

# load model, tokeniser?, and dataset

dsn = "amuvarma/luna-day3-classification"
tkn = "meta-llama/Llama-3.2-3B"
model_name = "amuvarma/llama-2.3m-full"
ds = load_dataset(dsn)
tokenizer = AutoTokenizer.from_pretrained(tkn)
tokenizer.add_special_tokens(
    {'additional_special_tokens': [f"[T{i}]" for i in range(9000)]})
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    # attn_implementation="flash_attention_2",
    num_labels=8,
)

ds = ds.shuffle(seed=42)

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




wandb.init(
    project="emotion-classification",
    name = "p0-15-11"
    )
 


training_args = TrainingArguments(
    per_device_train_batch_size=1,
    logging_steps=10,
    save_steps=0,
    report_to="wandb",
    evaluation_strategy="no",
    fp16=True,
    fsdp = "auto_wrap",
    output_dir=f"./{base_repo_id}",
)

# Initialize Trainer
trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],

)

# Train the model
print("Starting training...")
trainer.train()

full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = trainer.model.state_dict()

trainer.model.save_pretrained(f"./complete_{base_repo_id}", state_dict=state_dict)



