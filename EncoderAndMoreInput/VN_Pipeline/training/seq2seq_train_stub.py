"""
Stub for training an encoder-decoder on copy/replace tasks.

Use `EncoderAndMoreInput/encoder_decoder_backup.py` as the model reference.
This file should eventually train a seq2seq model that:
  - reads a source line with <hl>...</hl>
  - outputs the full line with the highlight replaced

Note: treat <hl> tokens as special tokens in the tokenizer, not literal text.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
import torch
from torch.nn import functional as F

repo_root = Path(__file__).resolve().parents[3]  # training/ -> VN_Pipeline/ -> EncoderAndMoreInput/ -> Chuni/
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from EncoderAndMoreInput.encoder_decoder_backup import EncoderDecoder
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding

def load_train_chunni_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    train_chunni_path = repo_root / "train-chunni.py"
    spec = importlib.util.spec_from_file_location("train_chunni", train_chunni_path)
    train_chunni = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_chunni)
    return train_chunni


def load_seq2seq_model() -> Any:
    """
    TODO: Instantiate EncoderDecoder with the same vocab and block size
    as your fine-tuned language model.
    - Optionally initialize weights from the fine-tuned LM.
    - Keep vocab_size consistent (50304 if using GPT-2 + special tokens).
    """
    config_path = Path(__file__).resolve().parents[3] / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    vocab_size = config.get("vocab_size", 50304)
    block_size = config.get("seq2seq_block_size", config.get("max_seq_len", 128))
    train_chunni = load_train_chunni_module()
    cfg = train_chunni.GPTConfig(vocab_size=vocab_size, block_size=block_size)
    model = EncoderDecoder(cfg, train_chunni.CausalSelfAttention)
    
    # Option to train from scratch (no pretrained init)
    if config.get("seq2seq_train_from_scratch", False):
        print("Training seq2seq from scratch (random init).", flush=True)
        return model
    
    init_path = resolve_vn_init_checkpoint(config)
    if init_path:
        try:
            gpt_state = load_gpt_state_dict(init_path)
            if init_seq2seq_from_gpt(model, gpt_state):
                print(f"Initialized seq2seq from VN checkpoint: {init_path}", flush=True)
            else:
                print("VN checkpoint found, but no compatible weights were loaded; using fresh init.", flush=True)
        except Exception as e:
            print(f"VN init failed ({e}); using fresh init.", flush=True)
    else:
        print("VN checkpoint not found; using fresh init.", flush=True)
    return model


def resolve_seq2seq_resume_path(config: dict, output_dir: str) -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []
    for key in ("seq2seq_resume_path", "seq2seq_checkpoint_path"):
        value = config.get(key)
        if value:
            candidates.append(Path(value))
    if output_dir:
        out_dir = Path(output_dir)
        candidates.append(out_dir / "model_final.pt")
        model_steps = []
        for path in out_dir.glob("model_*.pt"):
            match = re.search(r"model_(\d+)\.pt$", path.name)
            if match:
                model_steps.append((int(match.group(1)), path))
        if model_steps:
            model_steps.sort(key=lambda item: item[0])
            candidates.append(model_steps[-1][1])
    for path in candidates:
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            return path
    return None


def load_seq2seq_state_dict(checkpoint_path: Path) -> dict:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def load_seq2seq_best_val(checkpoint_path: Path) -> float | None:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        val = state.get("val_loss")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def resolve_vn_init_checkpoint(config: dict) -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    direct_path = config.get("seq2seq_init_checkpoint_path")
    if direct_path:
        candidates.append(Path(direct_path))
    resume_path = config.get("finetune_resume_path")
    if resume_path:
        candidates.append(Path(resume_path))
    finetune_dir = config.get("finetune_output_dir")
    if finetune_dir:
        finetune_dir = Path(finetune_dir)
        candidates.append(finetune_dir / "best_checkpoint.pt")
        candidates.extend(sorted(finetune_dir.glob("run_*/best_checkpoint.pt")))
    # Also check base GPT checkpoint as fallback
    base_path = config.get("base_checkpoint_path")
    if base_path:
        candidates.append(Path(base_path))
    for path in candidates:
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            return path
    return None


def load_gpt_state_dict(checkpoint_path: Path) -> dict:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            train_chunni = load_train_chunni_module()
            with torch.serialization.safe_globals([train_chunni.GPTConfig]):
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def init_seq2seq_from_gpt(model: EncoderDecoder, gpt_state: dict) -> bool:
    seq_state = model.state_dict()
    seq_keys = set(seq_state.keys())
    mapped = {}
    def maybe_assign(dst_key: str, src_key: str) -> None:
        if dst_key not in seq_keys or src_key not in gpt_state:
            return
        if seq_state[dst_key].shape != gpt_state[src_key].shape:
            return
        mapped[dst_key] = gpt_state[src_key]

    wte_key = "transformer.wte.weight"
    if wte_key not in gpt_state and "lm_head.weight" in gpt_state:
        wte_key = "lm_head.weight"
    maybe_assign("encoder.transformer.wte.weight", wte_key)
    maybe_assign("decoder.wte.weight", wte_key)
    maybe_assign("lm_head.weight", wte_key)
    maybe_assign("encoder.transformer.wpe.weight", "transformer.wpe.weight")
    maybe_assign("decoder.wpe.weight", "transformer.wpe.weight")
    maybe_assign("encoder.transformer.ln_f.weight", "transformer.ln_f.weight")
    maybe_assign("encoder.transformer.ln_f.bias", "transformer.ln_f.bias")
    maybe_assign("decoder.ln_f.weight", "transformer.ln_f.weight")
    maybe_assign("decoder.ln_f.bias", "transformer.ln_f.bias")

    for key in gpt_state:
        if not key.startswith("transformer.h."):
            continue
        enc_key = "encoder.transformer." + key[len("transformer."):]
        dec_key = "decoder." + key[len("transformer."):]
        maybe_assign(enc_key, key)
        maybe_assign(dec_key, key)

    if not mapped:
        return False
    model.load_state_dict(mapped, strict=False)
    return True

class Seq2SeqDataLoader:
    """
    Iterable dataloader for seq2seq training that supports multiple epochs.
    
    Unlike a generator, this class can be reset and iterated multiple times.
    It also shuffles data each epoch for better training.
    """
    
    def __init__(self, pairs_path: str, batch_size: int, seq_len: int, shuffle: bool = True):
        self.pairs_path = self._resolve_path(pairs_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.enc = build_hl_encoding()
        self.pad_id = self.enc.eot_token
        # Use a DIFFERENT token for BOS to distinguish from padding
        # Token 50256 is the original GPT-2 EOT, 50257/50258 are <hl>/</hl>
        # Use a high token that won't conflict (maps to rare token in vocab)
        self.bos_id = 50259  # Dedicated BOS token
        
        # Load and tokenize all pairs once
        self.pairs = self._load_pairs()
        self._indices = list(range(len(self.pairs)))
        self._pos = 0
    
    def _resolve_path(self, pairs_path: str) -> Path:
        pairs_file = Path(pairs_path)
        if pairs_file.is_file():
            return pairs_file
        if not pairs_file.is_absolute():
            alt_root = repo_root / pairs_file
            alt_nested = repo_root / "EncoderAndMoreInput" / "VN_Pipeline" / pairs_file
            if alt_root.is_file():
                return alt_root
            elif alt_nested.is_file():
                return alt_nested
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    
    def _load_pairs(self) -> list:
        """Load and tokenize all pairs from JSONL file."""
        pairs = []
        with open(self.pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                source = pair.get("source", "")
                target = pair.get("target", "")
                if not source or not target:
                    continue
                enc_tokens = self.enc.encode(source, allowed_special={"<hl>", "</hl>"})
                target_tokens = self.enc.encode(target)
                if not target_tokens:
                    continue
                pairs.append((enc_tokens, target_tokens))
        return pairs
    
    def _pad_to(self, tokens: list, length: int) -> list:
        if len(tokens) >= length:
            return tokens[:length]
        return tokens + [self.pad_id] * (length - len(tokens))
    
    def reset(self, shuffle: bool = None):
        """Reset to beginning of data. Optionally shuffle."""
        self._pos = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            import random
            random.shuffle(self._indices)
    
    def __len__(self):
        """Number of batches per epoch."""
        return len(self.pairs) // self.batch_size
    
    def __iter__(self):
        """Iterate through all batches once (one epoch)."""
        self.reset()
        while self._pos + self.batch_size <= len(self.pairs):
            batch_enc = []
            batch_dec_in = []
            batch_dec_out = []
            for i in range(self.batch_size):
                idx = self._indices[self._pos + i]
                enc_tokens, target_tokens = self.pairs[idx]
                decoder_input = [self.bos_id] + target_tokens[:-1]
                decoder_label = target_tokens
                batch_enc.append(self._pad_to(enc_tokens, self.seq_len))
                batch_dec_in.append(self._pad_to(decoder_input, self.seq_len))
                batch_dec_out.append(self._pad_to(decoder_label, self.seq_len))
            self._pos += self.batch_size
            yield (
                torch.tensor(batch_enc, dtype=torch.long),
                torch.tensor(batch_dec_in, dtype=torch.long),
                torch.tensor(batch_dec_out, dtype=torch.long),
            )
    
    def next_batch(self):
        """Get next batch, wrapping around to start if exhausted (infinite iteration)."""
        if self._pos + self.batch_size > len(self.pairs):
            self.reset()
        batch_enc = []
        batch_dec_in = []
        batch_dec_out = []
        for i in range(self.batch_size):
            idx = self._indices[self._pos + i]
            enc_tokens, target_tokens = self.pairs[idx]
            decoder_input = [self.bos_id] + target_tokens[:-1]
            decoder_label = target_tokens
            batch_enc.append(self._pad_to(enc_tokens, self.seq_len))
            batch_dec_in.append(self._pad_to(decoder_input, self.seq_len))
            batch_dec_out.append(self._pad_to(decoder_label, self.seq_len))
        self._pos += self.batch_size
        return (
            torch.tensor(batch_enc, dtype=torch.long),
            torch.tensor(batch_dec_in, dtype=torch.long),
            torch.tensor(batch_dec_out, dtype=torch.long),
        )


def build_seq2seq_dataloader(pairs_path: str, batch_size: int, seq_len: int, shuffle: bool = True) -> Seq2SeqDataLoader:
    """
    Load (source, target) pairs and build encoder/decoder batches.
    
    Returns a Seq2SeqDataLoader that can be iterated multiple times (multiple epochs).
    Use next_batch() for infinite iteration, or iterate directly for one epoch.
    """
    return Seq2SeqDataLoader(pairs_path, batch_size, seq_len, shuffle=shuffle)
    


def train_seq2seq_loop(
    model: Any,
    dataloader: Any,
    output_dir: str,
    val_pairs_path: str | None,
    batch_size: int,
    seq_len: int,
    max_steps: int,
    val_steps: int,
    val_interval: int,
    save_interval: int,
    best_val_init: float | None,
    learning_rate: float,
    start_step: int = 0,
) -> None:
    """
    Implement encoder-decoder training loop with teacher forcing.
    - Use cross-entropy on decoder outputs.
    - Save checkpoints to output_dir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model.to(device)
    enc = build_hl_encoding()
    pad_id = enc.eot_token
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics CSV for plotting
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w") as f:
            f.write("step,timestamp,train_loss,val_loss,avg_len_ratio,copy_match_rate,exact_match_rate\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
    step = start_step
    if start_step > 0:
        print(f"Resuming from step {start_step}", flush=True)
    best_val_loss = best_val_init if best_val_init is not None else float("inf")
    val_loader = None
    if val_pairs_path:
        val_loader = build_seq2seq_dataloader(val_pairs_path, batch_size, seq_len, shuffle=False)
    val_interval = max(1, val_interval)
    save_interval = max(1, save_interval)
    
    # Use next_batch() for infinite iteration through training data
    while step < max_steps:
        enc_in, dec_in, dec_out = dataloader.next_batch()
        model.train()
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        dec_out = dec_out.to(device)
        logits, _ = model(enc_in, dec_in)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            dec_out.view(-1),
            ignore_index=pad_id,
        )
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_non_pad = int((dec_out != pad_id).sum().item())
        val_loss = None
        val_non_pad_avg = None
        if val_loader and step % val_interval == 0:
            model.eval()
            val_loss_total = 0.0
            val_non_pad_total = 0
            for _ in range(max(1, val_steps)):
                val_enc, val_dec_in, val_dec_out = val_loader.next_batch()
                with torch.no_grad():
                    val_enc = val_enc.to(device)
                    val_dec_in = val_dec_in.to(device)
                    val_dec_out = val_dec_out.to(device)
                    val_logits, _ = model(val_enc, val_dec_in)
                    val_loss_tensor = F.cross_entropy(
                        val_logits.view(-1, val_logits.size(-1)),
                        val_dec_out.view(-1),
                        ignore_index=pad_id,
                    )
                    val_loss_total += val_loss_tensor.item()
                    val_non_pad_total += int((val_dec_out != pad_id).sum().item())
            val_loss = val_loss_total / max(1, val_steps)
            val_non_pad_avg = val_non_pad_total / max(1, val_steps)
        if val_loss is None:
            print(
                f"Step {step} | Loss: {loss.item():.6f} | Train non-pad: {train_non_pad}",
                flush=True,
            )
        else:
            print(
                f"Step {step} | Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | "
                f"Train non-pad: {train_non_pad} | Val non-pad avg: {val_non_pad_avg:.1f}",
                flush=True,
            )
            # Log to CSV for plotting
            from datetime import datetime
            with open(metrics_csv_path, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{step},{timestamp},{loss.item()},{val_loss},,,\n")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    "model": model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(best_checkpoint, os.path.join(output_dir, "best_checkpoint.pt"))
        if step % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_{step:05d}.pt"))
        step += 1
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pt"))
    print(f"Model saved to {os.path.join(output_dir, 'model_final.pt')}")


def main() -> None:
    """
    TODO:
    - Load config.json from repo root.
    - Load or init seq2seq model.
    - Build dataloader from copy/replace pairs.
    - Train and save checkpoints.
    """
    config_path = Path(__file__).resolve().parents[3] / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model = load_seq2seq_model()
    best_val_init = None
    start_step = 0
    if not config.get("seq2seq_force_vn_init", False):
        resume_path = resolve_seq2seq_resume_path(config, config["seq2seq_output_dir"])
        if resume_path:
            try:
                state = load_seq2seq_state_dict(resume_path)
                model.load_state_dict(state, strict=False)
                best_val_init = load_seq2seq_best_val(resume_path)
                print(f"Resuming seq2seq from: {resume_path}", flush=True)
                if best_val_init is not None:
                    print(f"Loaded best val loss: {best_val_init:.6f}", flush=True)
                # Extract step number from checkpoint filename (e.g., model_01500.pt -> 1500)
                match = re.search(r"model_(\d+)\.pt$", str(resume_path))
                if match:
                    start_step = int(match.group(1))
                    print(f"Starting from step {start_step}", flush=True)
            except Exception as e:
                print(f"Seq2seq resume failed ({e}); training from init.", flush=True)
    pairs_path = config.get("pairs_path")
    if not pairs_path:
        pairs_path = Path(config["seq2seq_output_dir"]) / "train.jsonl"
    seq_len = config.get("seq2seq_seq_len", config.get("max_seq_len", 128))
    max_steps = int(config.get("seq2seq_max_steps", 300))
    val_steps = int(config.get("seq2seq_val_steps", 5))
    val_interval = int(config.get("seq2seq_val_interval", 50))
    save_interval = int(config.get("seq2seq_save_interval", 250))
    learning_rate = float(config.get("seq2seq_learning_rate", 5e-5))
    val_pairs_path = config.get("val_pairs_path")
    if not val_pairs_path:
        val_pairs_path = str(Path(pairs_path).with_name("val.jsonl"))
    dataloader = build_seq2seq_dataloader(str(pairs_path), config["batch_size"], seq_len)
    train_seq2seq_loop(
        model,
        dataloader,
        config["seq2seq_output_dir"],
        str(val_pairs_path),
        config["batch_size"],
        seq_len,
        max_steps,
        val_steps,
        val_interval,
        save_interval,
        best_val_init,
        learning_rate,
        start_step,
    )
    print(f"Model saved to {os.path.join(config['seq2seq_output_dir'], 'model_final.pt')}")


if __name__ == "__main__":
    main()
