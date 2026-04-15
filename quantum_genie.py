#!/usr/bin/env python3
# quantum_genie.py – The Autonomous Orchestrator
# Runs the entire EiDom build with zero manual steps.
# Usage: python quantum_genie.py

import os
import sys
import subprocess
import time
import shutil
import tempfile
import platform

# ------------------------------------------------------------------
# 1. ENVIRONMENT CHECK & PACKAGE INSTALLATION
# ------------------------------------------------------------------
def run_cmd(cmd, capture=False, check=False):
    print(f"🔧 Running: {cmd}")
    if capture:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        subprocess.run(cmd, shell=True, check=check)

def install_system_packages():
    system = platform.system().lower()
    if "termux" in platform.release().lower() or os.path.exists("/data/data/com.termux"):
        print("📱 Termux detected – installing via pkg")
        run_cmd("pkg update -y")
        run_cmd("pkg install -y python espeak sox")
    elif system == "linux":
        print("🐧 Linux detected – trying apt")
        run_cmd("sudo apt update", check=False)
        run_cmd("sudo apt install -y espeak sox python3-pip", check=False)
    else:
        print("⚠️ Unknown OS – please install espeak, sox, python manually")

def install_python_packages():
    print("📦 Installing Python packages: flask, numpy")
    run_cmd("pip install flask numpy")
    # For slicing, we also need torch & transformers – will install temporarily if needed
    if not os.path.exists("model_fragment") or not os.path.exists("vocab_map.json"):
        print("📦 Model artifacts missing – will install torch & transformers temporarily")
        run_cmd("pip install torch transformers")

# ------------------------------------------------------------------
# 2. SLICE THE MODEL IF NOT PRESENT
# ------------------------------------------------------------------
def run_slice_model():
    if os.path.exists("model_fragment") and os.path.exists("vocab_map.json"):
        print("✅ model_fragment/ and vocab_map.json already exist – skipping slicing")
        return
    print("🔪 Slicing model (this downloads DistilGPT2 and extracts 1/4 layers)...")
    # Write slice_model.py temporarily if not present
    if not os.path.exists("slice_model.py"):
        slice_code = '''import torch, json, os, numpy as np
from transformers import DistilGPT2Model, DistilGPT2Tokenizer
print("Loading DistilGPT2...")
model = DistilGPT2Model.from_pretrained("distilgpt2")
state = model.state_dict()
os.makedirs("model_fragment", exist_ok=True)
layers_to_keep = 3
new_state = {}
for key, tensor in state.items():
    if "transformer.h" in key:
        if int(key.split(".")[3]) < layers_to_keep:
            new_state[key] = tensor.numpy()
    else:
        new_state[key] = tensor.numpy()
for name, arr in new_state.items():
    np.save(f"model_fragment/{name.replace('.','_')}.npy", arr)
config = model.config
config.n_layer = layers_to_keep
with open("model_fragment/config.json", "w") as f:
    json.dump(config.__dict__, f)
tokenizer = DistilGPT2Tokenizer.from_pretrained("distilgpt2")
vocab = {k: tokenizer.vocab[k] for k in list(tokenizer.vocab.keys())[:5000]}
with open("vocab_map.json", "w") as f:
    json.dump(vocab, f)
print("✅ Slicing complete.")
'''
        with open("slice_model.py", "w") as f:
            f.write(slice_code)
    # Run the slicing script
    run_cmd("python slice_model.py")
    # Clean up temporary heavy libs (optional)
    # run_cmd("pip uninstall -y torch transformers")

# ------------------------------------------------------------------
# 3. LAUNCH THE TREASURE NODE WITH AUTO‑RESTART
# ------------------------------------------------------------------
def run_treasure_node():
    print("🌟 Launching Eidolon Treasure Node...")
    # Ensure the standalone script exists
    if not os.path.exists("eidolon_standalone.py"):
        print("❌ eidolon_standalone.py not found – please place it in the same directory.")
        sys.exit(1)
    # Run it with auto‑restart on crash
    while True:
        print("🚀 Starting eidolon_standalone.py (Flask app)...")
        proc = subprocess.Popen(["python", "eidolon_standalone.py"])
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\n🛑 Orchestrator stopped by user.")
            break
        print("⚠️ Eidolon process died – restarting in 5 seconds...")
        time.sleep(5)

# ------------------------------------------------------------------
# 4. MAIN ORCHESTRATION
# ------------------------------------------------------------------
def main():
    print("🧞 ========== QUANTUM GENIE – AUTONOMOUS EIDOM BUILD ==========")
    install_system_packages()
    install_python_packages()
    run_slice_model()
    run_treasure_node()

if __name__ "__main__":
    main()
