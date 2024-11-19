import torch
from model import GazelleLlama

# 1. Load the model
# If you have the state dict saved:
model = GazelleLlama()  # You need to define the model architecture first
model.load_state_dict(torch.load("gazelle-llama-output/model.pt"))
model.eval()  # Set to evaluation mode

print(model)