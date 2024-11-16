import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import wandb

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

wandb.init(
    project="emotion-classification",
    name = "p0-15-11"
    )
 


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    logging_steps=10,
    save_steps=0,
    report_to="wandb",
    evaluation_strategy="no",
    fp16=True,
    fsdp = "auto_wrap",
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
