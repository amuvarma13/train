import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

teacher_model_name = "canopylabs/orpheus-3b-0.1-ft"
student_model_name = "amuvarma/1b-tts-pretrain-checkpoint-108493-of-108493"

teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher.eval()  # Freeze teacher parameters

student = AutoModelForCausalLM.from_pretrained(student_model_name)
student.resize_token_embeddings(teacher.config.vocab_size)  # Resize student embeddings to match teacher's vocabulary size

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

raw_dataset = load_dataset("amuvarma/voice-actors-13-full-audio3k-24k-notnormalised-dedup-TTS", split="train")

class PreTokenizedDataset(Dataset):
    def __init__(self, hf_dataset, pad_token_id, label_from_input_ids=True):
        self.input_ids = hf_dataset["input_ids"]
        self.pad_token_id = pad_token_id
        self.label_from_input_ids = label_from_input_ids

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx])
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone() if self.label_from_input_ids else None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.input_ids)

dataset = PreTokenizedDataset(raw_dataset, pad_token_id)

# Define a custom Trainer by subclassing the Hugging Face Trainer
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        device = model.device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        print("calculating teacher logits")
        
        teacher.to(student.device)
        print(teacher.device, input_ids.device, model.device)


        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        print("calculating student logits")

        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Compute distillation loss
        print("calculating distillation loss")  

        temperature = 2.0
        student_logits_temp = student_logits / temperature
        teacher_logits_temp = teacher_logits / temperature

        kd_loss = F.kl_div(
            F.log_softmax(student_logits_temp, dim=-1),
            F.softmax(teacher_logits_temp, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        loss = kd_loss
        return loss

# Define training arguments with a per-device batch size of 1.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Batch size of 1 as required.
    logging_steps=1,
    save_steps=50,
)

# Instantiate the custom trainer.
trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
