import torch
import os
from transformers import AutoTokenizer

print("cpu count", os.cpu_count())
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_dataset(ds):
    # Remove unnecessary columns
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["transcript", "audio"]])

    # Pre-compute instruction tokens once
    instruction = "Tell me the exact phrase I say back, my phrase is: "
    instruction_input_ids = tokenizer(instruction)["input_ids"]

    post_instruction = "The phrase you said is: "
    post_instruction_input_ids = tokenizer(post_instruction)["input_ids"]
    
    def process_batch(examples):
        batch_size = len(examples["transcript"])
        
        return {
            "audio_values": [torch.tensor(audio["array"]) for audio in examples["audio"]],
            "transcript_ids": tokenizer(examples["transcript"])["input_ids"],
            "labels": tokenizer(examples["transcript"])["input_ids"],
            "input_ids": [instruction_input_ids] * batch_size,
            "post_instruction": [post_instruction_input_ids] * batch_size
        }

    # Add batched=True and increase num_proc based on CPU count
    ds = ds.map(
        process_batch,
        batched=True,
        batch_size=100,  # Adjust based on memory constraints
        num_proc=max(1, os.cpu_count() - 1),  # Leave one CPU core free
        remove_columns=ds.column_names
    )
    
    # Keep only necessary columns
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["audio_values", "transcript_ids", "labels", "input_ids"]])
    
    return ds