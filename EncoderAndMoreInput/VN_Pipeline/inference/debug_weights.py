"""Debug weight loading - check checkpoint structures."""
import torch
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

print("="*70)
print("CHECKPOINT STRUCTURE DEBUG")
print("="*70)

# Check base GPT checkpoint
base_path = repo_root / "log/model_04999_clean.pt"
if base_path.exists():
    print(f"\n[1] Base GPT checkpoint: {base_path.name}")
    print(f"    Size: {base_path.stat().st_size / 1e6:.1f} MB")
    state = torch.load(base_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model" in state:
            model_state = state["model"]
            print(f"    Type: dict with 'model' key")
        else:
            model_state = state
            print(f"    Type: dict (direct state_dict)")
        print(f"    Keys: {len(model_state)} parameters")
        sample_keys = list(model_state.keys())[:5]
        print(f"    Sample keys: {sample_keys}")
        # Check for typical GPT keys
        has_transformer = any("transformer" in k for k in model_state.keys())
        has_wte = any("wte" in k for k in model_state.keys())
        print(f"    Has 'transformer' keys: {has_transformer}")
        print(f"    Has 'wte' keys: {has_wte}")
else:
    print(f"\n[1] Base GPT checkpoint NOT FOUND: {base_path}")

# Check VN finetune checkpoint
finetune_path = repo_root / "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs/best_checkpoint.pt"
if finetune_path.exists():
    print(f"\n[2] VN Finetune checkpoint: {finetune_path.name}")
    print(f"    Size: {finetune_path.stat().st_size / 1e6:.1f} MB")
    state = torch.load(finetune_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model" in state:
            model_state = state["model"]
            print(f"    Type: dict with 'model' key")
            # Check for other metadata
            for key in state.keys():
                if key != "model":
                    print(f"    Metadata key: {key} = {state[key] if not isinstance(state[key], dict) else '...'}")
        else:
            model_state = state
            print(f"    Type: dict (direct state_dict)")
        print(f"    Keys: {len(model_state)} parameters")
        sample_keys = list(model_state.keys())[:5]
        print(f"    Sample keys: {sample_keys}")
        # Check for typical GPT keys
        has_transformer = any("transformer" in k for k in model_state.keys())
        has_wte = any("wte" in k for k in model_state.keys())
        print(f"    Has 'transformer' keys: {has_transformer}")
        print(f"    Has 'wte' keys: {has_wte}")
else:
    print(f"\n[2] VN Finetune checkpoint NOT FOUND: {finetune_path}")

# Now test the actual weight loading
print("\n" + "="*70)
print("TESTING WEIGHT LOADING")
print("="*70)

from EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub import (
    load_seq2seq_model,
    resolve_vn_init_checkpoint,
    load_gpt_state_dict,
    init_seq2seq_from_gpt,
)
import json

config_path = repo_root / "config.json"
config = json.loads(config_path.read_text())

print(f"\nConfig settings:")
print(f"  seq2seq_train_from_scratch: {config.get('seq2seq_train_from_scratch', False)}")
print(f"  seq2seq_force_vn_init: {config.get('seq2seq_force_vn_init', False)}")
print(f"  finetune_output_dir: {config.get('finetune_output_dir')}")
print(f"  base_checkpoint_path: {config.get('base_checkpoint_path')}")

# Test resolve_vn_init_checkpoint
init_path = resolve_vn_init_checkpoint(config)
print(f"\nresolved init_path: {init_path}")

if init_path:
    print(f"\nTrying to load GPT state dict...")
    try:
        gpt_state = load_gpt_state_dict(init_path)
        print(f"  SUCCESS: Loaded {len(gpt_state)} keys")
        print(f"  Sample keys: {list(gpt_state.keys())[:5]}")

        # Create model and try to init
        from EncoderAndMoreInput.encoder_decoder_backup import EncoderDecoder
        import importlib.util
        train_chunni_path = repo_root / "train-chunni.py"
        spec = importlib.util.spec_from_file_location("train_chunni", train_chunni_path)
        train_chunni = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_chunni)

        vocab_size = config.get("vocab_size", 50304)
        block_size = config.get("seq2seq_block_size", config.get("max_seq_len", 128))
        cfg = train_chunni.GPTConfig(vocab_size=vocab_size, block_size=block_size)
        model = EncoderDecoder(cfg, train_chunni.CausalSelfAttention)

        print(f"\nEncoderDecoder model created:")
        print(f"  vocab_size: {vocab_size}")
        print(f"  block_size: {block_size}")

        # Check shapes
        seq_state = model.state_dict()
        print(f"\nShape comparison for key weights:")
        keys_to_check = ["encoder.transformer.wte.weight", "decoder.wte.weight"]
        gpt_wte_key = "transformer.wte.weight" if "transformer.wte.weight" in gpt_state else "lm_head.weight"
        for key in keys_to_check:
            if key in seq_state:
                seq_shape = seq_state[key].shape
                gpt_shape = gpt_state.get(gpt_wte_key, torch.zeros(1)).shape
                match = seq_shape == gpt_shape
                print(f"  {key}: seq={seq_shape}, gpt={gpt_shape}, match={match}")

        # Try the init
        success = init_seq2seq_from_gpt(model, gpt_state)
        print(f"\ninit_seq2seq_from_gpt returned: {success}")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nNo init_path found - model would train from scratch!")

print("\n" + "="*70)
