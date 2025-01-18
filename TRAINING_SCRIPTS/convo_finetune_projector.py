from datasets import load_dataset, concatenate_datasets
import wandb
import torch
import transformers
from transformers import Trainer, TrainingArguments
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import yaml
from tqdm import tqdm
from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
)
import whisper

whisper_model = whisper.load_model("small")
whisper_model = whisper_model.to("cuda:1")


config_file = "CONVO_FINETUNE_PROJECTOR_ARGS.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsns = config["dataset_names"]

model_name = config["model_name"]
base_model_name =  config["base_model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
learning_rate = config["learning_rate"]
vocab_size = config["vocab_size"]
audio_token_index = config["audio_token_index"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
audio_model_id = config["audio_model_id"]
audio_processor_id = config["audio_processor_id"] 
save_folder = config["save_folder"]
batch_size = config["batch_size"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]


MODEL_FOR_CAUSAL_LM_MAPPING.register(
    "gazelle", GazelleForConditionalGeneration)



config = GazelleConfig(
    audio_model_id=audio_model_id,
    text_model_id=model_name,
    audio_token_index=audio_token_index,
    vocab_size=vocab_size,

)
print("1")
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})

print("2")

config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=base_model_name,
    audio_token_index=audio_token_index,
    vocab_size=vocab_size,
)

base_model = GazelleForConditionalGeneration(config)

base_model.resize_token_embeddings(len(tokenizer))
special_config =  base_model.config
print("3")

model = GazelleForConditionalGeneration.from_pretrained(model_name, config=special_config, new_vocab_size=True)
print("4")
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
# for name, param in model.named_parameters():
#     if "language_model" in name:
#         param.requires_grad = True


wandb.init(project=project_name, name=run_name)

all_datasets = []
for dsn in dsns:
    fractional_dataset = load_dataset(dsn, split="train")
    all_datasets.append(fractional_dataset)

dataset = concatenate_datasets(all_datasets)
dataset = dataset.shuffle(seed=42)
print("5")
def remove_short_audio(dataset, min_seconds=1.0):
    indices_to_keep = []

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        audio = example['question_audio']
        duration = len(audio['array']) / audio['sampling_rate']
        if duration >= min_seconds:
            indices_to_keep.append(i)

    filtered_dataset = dataset.select(indices_to_keep)

    return filtered_dataset

# dataset = remove_short_audio(dataset)

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    audio_processor_id)


def process_audio_tensor(audio, sample_rate=16000):
    audio = audio.to(torch.float32)
    duration_ms = (len(audio) / sample_rate) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1

def pad_and_cat_sequences(sequence_list):
    max_len = max(t.shape[1] for t in sequence_list)
    padded_features = []
    lengths = []
    
    for t in sequence_list:
        curr_len = t.shape[1]
        lengths.append(curr_len)
        pad_len = max_len - curr_len
        padded = torch.nn.functional.pad(t, (0, 0, 0, pad_len, 0, 0))
        padded_features.append(padded)

    return torch.cat(padded_features), torch.tensor(lengths)

def inference_collator(audios, input_ids, labels, attention_mask):

    max_len = max(len(seq) for seq in audios)
    padded_sequences = []
    for seq in audios:
        padded = seq + [0] * (max_len - len(seq))  # Zero padding
        padded_sequences.append(padded)

    processed_features = []
    for audio_input in audios:
        #convert audio input to tensor
        audio_input = torch.tensor(audio_input)
        audio_input = audio_input.squeeze(0)
        mel, length = process_audio_tensor(audio_input)
        mel = mel.to(whisper_model.device)
        mel = mel.unsqueeze(0)
        audio_feature = whisper_model.embed_audio(mel)[0][:length]
        audio_feature = audio_feature.unsqueeze(0)
        processed_features.append(audio_feature)
    padded_audio, lengths = pad_and_cat_sequences(processed_features)
    print("audio feature size", padded_audio.shape)
    print("lengths", lengths)

    #convert input_ids, labels, attention_mask to tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    labels = torch.tensor(labels).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    print("shapes", input_ids.shape, labels.shape, attention_mask.shape)

    return {
        "audio_values": padded_audio.to(model.device).to(model.dtype),
        "input_ids": input_ids.to(model.device),
        "labels": labels.to(model.device),
        "attention_mask": attention_mask.to(model.device),
    }



class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):
        input_ids = features[0]["input_ids"]
        attention_mask = features[0]["attention_mask"]
        labels = features[0]["labels"]
        audios = features[0]["audios"]

        batch = inference_collator(audios, input_ids, labels, attention_mask)
        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }


training_args = TrainingArguments(
    output_dir=save_folder,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,
    learning_rate=learning_rate,  # Changed to 2*10^-3
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=15000
)

print("7")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)


trainer.train()
