import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Step 1: Load the model and tokenizer
model_name = "amuvarma/convo-llama-13k-text"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Load the dataset
dataset = load_dataset("amuvarma/audio_eval_small")

# Step 3: Create DataLoader
dataloader = DataLoader(dataset["train"], batch_size=1)

# Step 4: Compute the loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

total_loss = 0
total_samples = 0

with torch.no_grad():
    for batch in dataloader:
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)

average_loss = total_loss / total_samples
print(f"Average loss: {average_loss:.4f}")