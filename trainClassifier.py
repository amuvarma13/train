import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset


#load model, tokeniser?, and dataset

dsn  = "amuvarma/luna-day3-classification"
tkn = "meta-llama/Llama-3.2-3B"
model_name = "amuvarma/llama-2.3m-full"
ds = load_dataset(dsn)
tokenizer = AutoTokenizer.from_pretrained(tkn)
tokenizer.add_special_tokens({'additional_special_tokens': [f"[T{i}]" for i in range(9000)]})
model = AutoModelForSequenceClassification.from_pretrained(model_name)

ds = ds.shuffle(seed=42)   


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_steps=0,
    evaluation_strategy="no",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
)

# Train the model
print("Starting training...")
trainer.train()

