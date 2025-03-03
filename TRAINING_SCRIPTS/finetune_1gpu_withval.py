import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "FINETUNE_ARGS-3b-ztts.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]
resize_dataset = config["resize_dataset"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

# Initialize tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# Load dataset - if it doesn't have a validation split, do train_test_split manually
dataset = load_dataset(dsn, split="train")
max_len = max(len(row["input_ids"]) for row in dataset)
print("max_len", max_len)

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

wandb.init(project=project_name, name=run_name)

# Optionally add new tokens
if resize_dataset:
    number_add_tokens = 7 * 4096 + 10
    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

# Shuffle the train dataset if needed
train_dataset = train_dataset.shuffle(seed=42)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy} 

def data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    print(len(input_ids))
    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    # Convert all lists to tensors and pad
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb",
    save_steps=save_steps,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,      # Pass the validation set here
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()
