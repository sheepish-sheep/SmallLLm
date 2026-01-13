"""Debug what weights are actually being transferred."""
import torch
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import json
import importlib.util

# Load config
config_path = repo_root / "config.json"
config = json.loads(config_path.read_text())

# Load train_chunni module
train_chunni_path = repo_root / "train-chunni.py"
spec = importlib.util.spec_from_file_location("train_chunni", train_chunni_path)
train_chunni = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_chunni)

# Create EncoderDecoder model
from EncoderAndMoreInput.encoder_decoder_backup import EncoderDecoder

vocab_size = config.get("vocab_size", 50304)
block_size = config.get("seq2seq_block_size", config.get("max_seq_len", 128))
cfg = train_chunni.GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = EncoderDecoder(cfg, train_chunni.CausalSelfAttention)

# Load VN finetune checkpoint
finetune_path = repo_root / "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs/best_checkpoint.pt"
state = torch.load(finetune_path, map_location="cpu", weights_only=False)
gpt_state = state["model"] if "model" in state else state

print("="*70)
print("WEIGHT MAPPING ANALYSIS")
print("="*70)

seq_state = model.state_dict()
seq_keys = set(seq_state.keys())

print(f"\nEncoderDecoder has {len(seq_keys)} parameters")
print(f"GPT checkpoint has {len(gpt_state)} parameters")

# Try to map weights
mapped = {}
not_mapped_seq = []
not_mapped_gpt = []

def maybe_assign(dst_key: str, src_key: str) -> bool:
    if dst_key not in seq_keys:
        return False
    if src_key not in gpt_state:
        return False
    if seq_state[dst_key].shape != gpt_state[src_key].shape:
        print(f"  SHAPE MISMATCH: {dst_key} {seq_state[dst_key].shape} vs {src_key} {gpt_state[src_key].shape}")
        return False
    mapped[dst_key] = src_key
    return True

# Map embeddings
wte_key = "transformer.wte.weight" if "transformer.wte.weight" in gpt_state else "lm_head.weight"
maybe_assign("encoder.transformer.wte.weight", wte_key)
maybe_assign("decoder.wte.weight", wte_key)
maybe_assign("lm_head.weight", wte_key)
maybe_assign("encoder.transformer.wpe.weight", "transformer.wpe.weight")
maybe_assign("decoder.wpe.weight", "transformer.wpe.weight")
maybe_assign("encoder.transformer.ln_f.weight", "transformer.ln_f.weight")
maybe_assign("encoder.transformer.ln_f.bias", "transformer.ln_f.bias")
maybe_assign("decoder.ln_f.weight", "transformer.ln_f.weight")
maybe_assign("decoder.ln_f.bias", "transformer.ln_f.bias")

# Map transformer blocks
for key in gpt_state:
    if not key.startswith("transformer.h."):
        continue
    enc_key = "encoder.transformer." + key[len("transformer."):]
    dec_key = "decoder." + key[len("transformer."):]
    maybe_assign(enc_key, key)
    maybe_assign(dec_key, key)

# Find what's NOT mapped
for key in seq_keys:
    if key not in mapped:
        not_mapped_seq.append(key)

for key in gpt_state:
    if key not in [v for v in mapped.values()]:
        not_mapped_gpt.append(key)

print(f"\nSuccessfully mapped: {len(mapped)} / {len(seq_keys)} EncoderDecoder params")
print(f"Not mapped from EncoderDecoder: {len(not_mapped_seq)}")
print(f"Not used from GPT: {len(not_mapped_gpt)}")

if not_mapped_seq:
    print(f"\nEncoderDecoder params NOT receiving pretrained weights:")
    for key in sorted(not_mapped_seq)[:20]:
        print(f"  - {key}")
    if len(not_mapped_seq) > 20:
        print(f"  ... and {len(not_mapped_seq) - 20} more")

# Calculate what % of parameters are initialized
total_params = sum(p.numel() for p in model.parameters())
mapped_params = sum(seq_state[k].numel() for k in mapped.keys())
pct = 100 * mapped_params / total_params
print(f"\n{pct:.1f}% of parameters initialized from pretrained ({mapped_params:,} / {total_params:,})")

print("\n" + "="*70)
