from datasets import load_dataset, concatenate_datasets
import wandb
import torch
import transformers
from transformers import Trainer, TrainingArguments
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import yaml
from orpheus import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
config = OrpheusConfig(
            text_model_id=model_name,
            audio_token_index=156939,
            vocab_size=156939,
            hidden_size=3072,
        )

model = OrpheusForConditionalGeneration(config)

print(model)

