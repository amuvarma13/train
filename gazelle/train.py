from transformers import Trainer, TrainingArguments

from model import GazelleLlama
from datasets import load_dataset
from preprocess_dataset import preprocess_dataset
import wandb

dsn = "parler-tts/mls_eng_10k"
gazelle_model = GazelleLlama()
gazelle_model = gazelle_model.cuda()

ds = load_dataset(dsn)

dsp = preprocess_dataset(ds["dev"])


project_name = "gazelle-projection"
wandb.init(
    project=project_name,
    name = "p0-17-11-dev-b2-gpu2"
)
 
 



# Define minimal training arguments with wandb disabled
training_args = TrainingArguments(
    output_dir="./gazelle-llama-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    report_to="wandb",  # Disables wandb and other reporting
    save_safetensors=False, 
    logging_steps=100
)

# Initialize trainer
trainer = Trainer(
    model=gazelle_model,
    args=training_args,
    train_dataset=dsp,
)

# Start training
trainer.train()

