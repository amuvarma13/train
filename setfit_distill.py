from setfit import DistillationTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

teacher_model_name = "canopylabs/orpheus-3b-0.1-ft"
student_model_name = "amuvarma/1b-tts-pretrain-checkpoint-108493-of-108493"

teacher_model = SetFitModel.from_pretrained(teacher_model_name, attn_implementation="flash_attention_2")

model = SetFitModel.from_pretrained(student_model_name, attn_implementation="flash_attention_2")
teacher_model.resize_token_embeddings(model.config.vocab_size)  # Resize student embeddings to match teacher's vocabulary size

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

raw_dataset = load_dataset("amuvarma/em-EN-TTS-full-8192", split="train")


distillation_args = TrainingArguments(
    per_device_train_batch_size=16,
    max_steps=500,
)

distillation_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=model,
    args=distillation_args,
    train_dataset=raw_dataset,
)
# Train student with knowledge distillation
distillation_trainer.train()
