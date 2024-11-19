from transformers import Trainer, TrainingArguments

from model import GazelleLlama
from datasets import load_dataset
from preprocess_dataset import preprocess_dataset
import wandb

dsn = "amuvarma/mls-train-200-1000"
gazelle_model = GazelleLlama()
gazelle_model = gazelle_model.cuda()

ds = load_dataset(dsn)

# dsp = preprocess_dataset(ds["dev"])
dsp = ds["train"]

project_name = "gazelle-projection"
wandb.init(
    project=project_name,
    name = "p6-18-11-dev-b4-gpu8-save-test"
)
 
 



# Define minimal training arguments with wandb disabled
training_args = TrainingArguments(
    output_dir="./gazelle-llama-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=4e-3,
    report_to="wandb",  # Disables wandb and other reporting
    save_safetensors=False, 
    logging_steps=12, 
    # bf16=True,
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
gazelle_model.save_pretrained("gazelle-llama-output")

