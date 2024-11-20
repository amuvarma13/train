from transformers import Trainer, TrainingArguments

from model import GazelleLlama
from datasets import load_dataset
from preprocess_dataset import preprocess_dataset
import wandb
import torch

dsn = "amuvarma/mls-train-200"

gazelle_model = GazelleLlama()

ds = load_dataset(dsn)

# dsp = preprocess_dataset(ds["dev"])
dsp = ds["train"]

project_name = "gazelle-projection"
wandb.init(
    project=project_name,
    name = "p8-8gpu-3b-lossfn-n"
)
 
 



# Define minimal training arguments with wandb disabled
training_args = TrainingArguments(
    output_dir="./gazelle-llama-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=4e-3,
    report_to="wandb",  # Disables wandb and other reporting
    save_safetensors=False, 
    logging_steps=24,
    save_strategy="no",
)

# Initialize trainer
trainer = Trainer(
    model=gazelle_model,
    args=training_args,
    train_dataset=dsp,
)

# Start training
trainer.train() 

# Save model
torch.save(gazelle_model.state_dict(), "gazelle-llama-output/model.pt")

