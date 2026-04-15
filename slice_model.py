# slice_model.py
# Run this ONCE on a machine with torch and transformers installed.
# It creates the model_fragment/ folder and vocab_map.json.
# Then copy these two items to your Termux / Codespaces environment.

import torch
import json
import os
import numpy as np
from transformers import DistilGPT2Model, DistilGPT2Tokenizer

# ------------------------------------------------------------------
# 1. Slice the model: keep only first 3 layers (1/4 of 12 layers)
# ------------------------------------------------------------------
print("📥 Loading full DistilGPT2 model...")
model = DistilGPT2Model.from_pretrained("distilgpt2")
state = model.state_dict()

os.makedirs("model_fragment", exist_ok=True)

layers_to_keep = 3   # 3 out of 12 = 1/4 of the model
new_state = {}

for key, tensor in state.items():
    if "transformer.h" in key:
        # Extract layer number from key like "transformer.h.0.attn..."
        layer_num = int(key.split(".")[3])
        if layer_num < layers_to_keep:
            new_state[key] = tensor.numpy()
    else:
        # Keep embeddings, final layer norm, etc.
        new_state[key] = tensor.numpy()

# Save each weight as a separate .npy file
for name, arr in new_state.items():
    safe_name = name.replace(".", "_")
    np.save(f"model_fragment/{safe_name}.npy", arr)
    print(f"   Saved {safe_name}.npy")

# Save the model config with reduced n_layer
config = model.config
config.n_layer = layers_to_keep
with open("model_fragment/config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

print("✅ Model fragment saved to model_fragment/ (first 3 layers)\n")

# ------------------------------------------------------------------
# 2. Create a lightweight vocabulary mapping (first 5000 tokens)
# ------------------------------------------------------------------
print("📝 Creating vocabulary map (first 5000 tokens)...")
tokenizer = DistilGPT2Tokenizer.from_pretrained("distilgpt2")
vocab = {k: tokenizer.vocab[k] for k in list(tokenizer.vocab.keys())[:5000]}
with open("vocab_map.json", "w") as f:
    json.dump(vocab, f, indent=2)

print("✅ vocab_map.json saved (5000 tokens)")
print("\n🎉 Done! Now copy the 'model_fragment/' folder and 'vocab_map.json' to your target device.")
