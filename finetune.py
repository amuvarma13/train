from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

dsn = "amuvarma/24k-gcp-llama"
ds = load_dataset(dsn)



# Load the pre-trained model and tokenizer
model_name = "amuvarma/llama-checkpoint-180000"  # Replace with your model
model = AutoModelForCausalLM.from_pretrained(model_name)



# Define LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj" ]
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=20,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")