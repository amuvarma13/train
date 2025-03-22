import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import wandb

wandb.init(project="distilling-3b-dev-pre", name="r1-5e5-t2")

teacher_model_name = "canopylabs/orpheus-3b-0.1-pretrained"
student_model_name = "amuvarma/1b-tts-pretrain-checkpoint-108493-of-108493"

teacher = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, attn_implementation="flash_attention_2"
).to(torch.bfloat16)
teacher.eval() 

student = AutoModelForCausalLM.from_pretrained(
    student_model_name, attn_implementation="flash_attention_2"
)
teacher.resize_token_embeddings(student.config.vocab_size)  # Resize teacher embeddings to match student's vocabulary size

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

print("loading dataset")

raw_dataset = load_dataset("amuvarma/em-EN-TTS-full-8192", split="train")

print("loaded dataset")

class LazyPreTokenizedDataset(Dataset):
    def __init__(self, hf_dataset, pad_token_id, label_from_input_ids=True):
        # Instead of copying the whole column, store the dataset reference.
        self.hf_dataset = hf_dataset
        self.pad_token_id = pad_token_id
        self.label_from_input_ids = label_from_input_ids

    def __getitem__(self, idx):
        # Fetch the example on demand.
        example = self.hf_dataset[idx]
        input_ids = torch.tensor(example["input_ids"])
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone() if self.label_from_input_ids else None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.hf_dataset)


dataset = LazyPreTokenizedDataset(raw_dataset, pad_token_id)

# Define a custom Trainer by subclassing the Hugging Face Trainer
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        device = model.device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Ensure the teacher is on the same device as the student/model.
        teacher.to(device)

        # Compute teacher outputs (without gradients).
        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Compute student outputs.
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        temperature = 2.0
        student_logits_temp = student_logits / temperature
        teacher_logits_temp = teacher_logits / temperature

        kd_loss = F.kl_div(
            F.log_softmax(student_logits_temp, dim=-1),
            F.softmax(teacher_logits_temp, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        # Compute standard cross entropy losses.
        # Shift logits and labels for next-token prediction.
        shift_student_logits = student_logits[:, :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        student_ce_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id
        )
        teacher_ce_loss = F.cross_entropy(
            shift_teacher_logits.view(-1, shift_teacher_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id
        )

        if device == torch.device("cuda", 0):
            wandb.log({
                "kd_loss": kd_loss.item(),
                "student_ce_loss": student_ce_loss.item(),
                "teacher_ce_loss": teacher_ce_loss.item(), 
                "ce-diff": student_ce_loss.item() - teacher_ce_loss.item()
            })

        return kd_loss


print("initialized training args")
# Define training arguments with a per-device batch size of 1.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Batch size of 1 as required.
    logging_steps=1,
    save_steps=12000,
    report_to="wandb",
    learning_rate=5e-5,
    bf16=True,
    fsdp="auto_wrap",
)

print("initialized trainer")

# Instantiate the custom trainer.
trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
