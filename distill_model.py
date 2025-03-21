import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

teacher_model_name = "big-model"
student_model_name = "small-model"

teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher.eval() 

student = AutoModelForCausalLM.from_pretrained(student_model_name)

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# Example texts; replace these with your actual dataset.
texts = [
    "Hello, how are you?",
    "This is an example sentence for distillation.",
    "Knowledge distillation transfers teacher behavior into a smaller model."
]
dataset = TextDataset(texts, tokenizer)

# Custom loss function that trains the student solely on teacher outputs.
def compute_loss(model, inputs, return_outputs=False):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():
        teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits

    student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_outputs.logits

    # Temperature scaling
    temperature = 2.0
    student_logits_temp = student_logits / temperature
    teacher_logits_temp = teacher_logits / temperature

    kd_loss = F.kl_div(
        F.log_softmax(student_logits_temp, dim=-1),
        F.softmax(teacher_logits_temp, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    loss = kd_loss

    return (loss, student_outputs) if return_outputs else loss

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_steps=50,
    weight_decay=0.01,
)

# Create the Trainer with the custom loss function
trainer = Trainer(
    model=student,
    args=training_args,
    train_dataset=dataset,
    compute_loss=compute_loss,
)

# Start the distillation training process
trainer.train()
