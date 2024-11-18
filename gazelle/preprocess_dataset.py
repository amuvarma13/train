import torch

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

def pad_or_crop_audio(audio_tensor, target_length=300000):
    if len(audio_tensor) > target_length:
        return audio_tensor[:target_length]
    elif len(audio_tensor) < target_length:
        return torch.cat([audio_tensor, torch.zeros(target_length - len(audio_tensor))])
    return audio_tensor

def preprocess_dataset(ds, batch_size=64):
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["transcript", "audio"]])

    instruction = "Read out this phrase back."
    instruction_inputs = tokenizer(instruction)
    instruction_input_ids = instruction_inputs["input_ids"]

    def process_batch(examples):
        batch_size = len(examples["audio"])
        
        audio_values = [
            pad_or_crop_audio(torch.tensor(audio["array"]))
            for audio in examples["audio"]
        ]
        
        transcript_encodings = tokenizer(
            examples["transcript"],
            padding="max_length",
            max_length=100,
            truncation=True
        )
        
        return {
            "audio_values": audio_values,
            "transcript_ids": transcript_encodings["input_ids"],
            "labels": transcript_encodings["input_ids"],
            "input_ids": [instruction_input_ids] * batch_size
        }

    ds = ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names
    )
    
    return ds