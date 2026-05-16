import torch
import os
from pathlib import Path
from tokenizers import Tokenizer
from micro_transformer_v2 import MicroTransformer
from dataclasses import dataclass

@dataclass
class Config: 
    vocab_size: int = None

# --- DYNAMIC SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file("data/tokenizer.json")

def get_latest_checkpoint(ckpt_dir="checkpoints"):
    """Finds the .pt file with the highest step count."""
    ckpts = list(Path(ckpt_dir).glob("ckpt_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found! Did you move them?")
    # Sorts by the number in the filename
    return max(ckpts, key=lambda p: int(p.stem.split("_")[2]))

# Load the latest automatically
ckpt_path = get_latest_checkpoint()

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
print(checkpoint.keys())

model = MicroTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    seq_len=2048,
    dim=300,
    n_heads=10,
    n_layers=12,
    dropout=0.1
).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Walt Loaded Automatically: {ckpt_path.name}")

# --- SNIPPET LOGIC ---
def get_snippet(prompt, max_length=128):
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    
    with torch.no_grad():
        output_ids = model.generate(
            idx=input_ids,
            max_new_tokens=max_length,
            temperature=1,
            top_k=40,
            top_p=0.9,
            repetition_penalty=6.0
        )
    
    return tokenizer.decode(output_ids[0].tolist())

# --- THE LOOP ---
print("Welcome to WALT (Wicked Awesome Luau Transformer)! Type 'exit' or 'quit' to close.")

while True:
    try:
        user_prompt = input("\nEnter Luau Prompt: ")
        
        if user_prompt.lower() in ("exit", "quit"):
            print("Closing Walt.")
            break

        print("\n--- WALT SNIPPET ---")
        print(get_snippet(user_prompt))
        print("--------------------")

    except KeyboardInterrupt:
        print("\nCya!")
        break