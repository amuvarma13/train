import torch

# Load the optimizer state
optimizer_state = torch.load("./mymodel/checkpoint-200/optimizer.bin", map_location=torch.device('cpu'))

# Print the top-level keys
print("Top-level keys:")
for key in optimizer_state.keys():
    print(f"- {key}")

# If there's a 'state' key, which is common in optimizer states, inspect it further
if 'state' in optimizer_state:
    print("\nState dict keys:")
    for key in optimizer_state['state'].keys():
        print(f"- {key}")

    # Print the keys for the first state entry (if it exists)
    first_state_key = next(iter(optimizer_state['state']))
    print(f"\nKeys for state entry {first_state_key}:")
    for key in optimizer_state['state'][first_state_key].keys():
        print(f"- {key}")