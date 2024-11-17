from transformers import Trainer, TrainingArguments

from model import GazelleLlama
from datasets import load_dataset
from gazelle.preprocess_dataset import preprocess_dataset

dsn = "parler-tts/mls_eng_10k"
gazelle_model = GazelleLlama()

ds = load_dataset(dsn)

dsp = preprocess_dataset(ds)


# Define minimal training arguments with wandb disabled
training_args = TrainingArguments(
    output_dir="./gazelle-llama-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    report_to="none",  # Disables wandb and other reporting
    save_safetensors=False, 
    logging_steps=1
)

# Initialize trainer
trainer = Trainer(
    model=gazelle_model,
    args=training_args,
    train_dataset=dsp,
)

# Start training
trainer.train()
