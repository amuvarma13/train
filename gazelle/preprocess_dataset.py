import torch

def pad_or_crop_audio(audio_tensor, target_length=300000):
    if len(audio_tensor) > target_length:
        return audio_tensor[:target_length]
    elif len(audio_tensor) < target_length:
        return torch.cat([audio_tensor, torch.zeros(target_length - len(audio_tensor))])
    return audio_tensor


def preprocess_dataset(ds):
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["transcript", "audio"]])

    instruction = "Read out this phrase back."
    instruction_inputs = tokenizer(instruction, return_tensors="pt")
    instruction_input_ids = inputs["input_ids"]

    
    def process_example(example):
        example["audio_values"] = pad_or_crop_audio(torch.tensor(example["audio"]["array"], dtype=torch.float32))
        example["transcript_ids"] = tokenizer(example["transcript"])["input_ids"]
        example["labels"] = example["transcript_ids"]
        example["input_ids"] = instruction_input_ids[0]
        return example

    ds = ds.map(process_example)
    ds = ds.remove_columns([col for col in ds.column_names if col not in ["audio_values", "transcript_ids", "labels", "input_ids" ]])
    
    return ds


