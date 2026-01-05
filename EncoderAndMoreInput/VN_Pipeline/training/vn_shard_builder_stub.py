"""
Stub for building VN token shards (train/val) similar to fineweb.py.

Reads `training_data/vn/cleaned_binary_dialogue.txt`, tokenizes, splits
into train/val, and writes .npy shards.
"""

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding


def load_config(path: str = "config.json"):
    """
    TODO: read config.json from repo root.
    - Expect vn_cleaned_path and max_seq_len at minimum.
    - Return a dict-like config object.
    """
    with open(path, "r") as f:
        config = json.load(f)
    return config



def load_vn_text(path: str) -> str:
    """
    TODO: read the cleaned VN text file and return full text.
    - Use utf-8.
    - Optionally filter very short lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def build_encoding():
    """
    TODO: return the tokenizer/encoding you will use for VN data.
    - If you want a single tokenizer everywhere, use build_hl_encoding().
    - Otherwise, use tiktoken.get_encoding("gpt2") for plain VN text.
    """
    return build_hl_encoding()


def write_shards(tokens, output_dir: str, shard_size: int, split: str) -> None:
    """
    TODO: write `train_000.npy`, `val_000.npy`, etc.
    - Create output_dir if missing.
    - Decide a train/val split (ex: first 5% tokens to val).
    - Use np.save to write uint16 arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_index = 0
    for i in range(0, len(tokens), shard_size):
        shard = tokens[i:i + shard_size]
        shard_path = os.path.join(output_dir, f"{split}_{shard_index:06d}.npy")
        np.save(shard_path, np.asarray(shard, dtype=np.uint16))
        shard_index += 1


def main() -> None:
    """
    TODO:
    - load config.json
    - read VN text
    - tokenize with build_encoding()
    - split into train/val
    - write shards to a data directory
    """
    config = load_config()
    text = load_vn_text(config["vn_cleaned_path"])
    tokens = build_encoding().encode(text)
    split_idx = int(len(tokens) * 0.95)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    shard_dir = config.get("vn_shard_dir")
    if not shard_dir:
        raise ValueError("vn_shard_dir is required in config.json")
    shard_size = int(config.get("shard_size", 1_000_000))
    write_shards(train_tokens, shard_dir, shard_size, split="train")
    write_shards(val_tokens, shard_dir, shard_size, split="val")


if __name__ == "__main__":
    main()
