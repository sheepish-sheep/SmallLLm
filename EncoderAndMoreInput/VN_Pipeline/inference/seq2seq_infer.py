import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from EncoderAndMoreInput.encoder_decoder_backup import EncoderDecoder
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding


def load_train_chunni_module(repo_root: Path) -> Any:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    train_chunni_path = repo_root / "train-chunni.py"
    spec = importlib.util.spec_from_file_location("train_chunni", train_chunni_path)
    train_chunni = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_chunni)
    return train_chunni


def load_seq2seq_model(config: dict, repo_root: Path) -> Any:
    train_chunni = load_train_chunni_module(repo_root)
    vocab_size = config.get("vocab_size", 50304)
    block_size = config.get("seq2seq_block_size", config.get("max_seq_len", 128))
    cfg = train_chunni.GPTConfig(vocab_size=vocab_size, block_size=block_size)
    model = EncoderDecoder(cfg, train_chunni.CausalSelfAttention)
    return model


def load_checkpoint(model: Any, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Seq2seq checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)


def generate_replacement(
    text: str,
    model: Any,
    enc,
    max_len: int,
    temperature: float,
    top_k: int,
    device: str,
    source_prefix: str,
    source_suffix: str,
    repetition_penalty: float = 1.2,
) -> str:
    # No <hl> tags needed - we're doing full sentence replacement
    text = f"{source_prefix}{text}{source_suffix}"
    source_tokens = enc.encode(text)
    if len(source_tokens) > model.config.block_size:
        source_tokens = source_tokens[-model.config.block_size:]
    enc_in = torch.tensor(source_tokens, dtype=torch.long, device=device).unsqueeze(0)

    bos_id = 50259  # Must match training BOS token
    eos_id = enc.eot_token
    dec_tokens = [bos_id]

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            dec_in = torch.tensor(dec_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(enc_in, dec_in)
            logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(dec_tokens):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_id = indices.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            next_token = next_id.item()
            dec_tokens.append(next_token)
            if next_token == eos_id:
                break

    decoded = enc.decode(dec_tokens[1:])  # drop BOS
    return decoded.strip()


class Seq2SeqRewriter:
    """
    Seq2seq model wrapper for inference.
    
    Usage:
        rewriter = Seq2SeqRewriter(Path("config.json"))
        output = rewriter.rewrite("The <hl>quick</hl> brown fox")
    """
    
    def __init__(self, config_path: Path, repo_root: Path = None):
        """
        Initialize the rewriter from a config file.
        
        Args:
            config_path: Path to config.json
            repo_root: Optional explicit repo root. If not provided, uses the
                      standard location (2 directories up from VN_Pipeline).
        """
        # Determine repo root - don't assume config is in repo root
        if repo_root is None:
            # Default: assume this file is in EncoderAndMoreInput/VN_Pipeline/
            repo_root = Path(__file__).resolve().parents[3]
        self.repo_root = repo_root
        
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.enc = build_hl_encoding()
        self.source_prefix = self.config.get("seq2seq_source_prefix", "")
        self.source_suffix = self.config.get("seq2seq_source_suffix", "")
        self.len_ratio = float(self.config.get("seq2seq_len_ratio", 1.1))
        self.min_gen_len = int(self.config.get("seq2seq_min_gen_len", 8))
        self.repetition_penalty = float(self.config.get("seq2seq_repetition_penalty", 1.2))

        self.model = load_seq2seq_model(self.config, repo_root)
        checkpoint_path = self.config.get("seq2seq_checkpoint_path")
        if not checkpoint_path:
            checkpoint_path = Path(self.config["seq2seq_output_dir"]) / "model_final.pt"
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = repo_root / checkpoint_path
        load_checkpoint(self.model, checkpoint_path)

        device = self.config.get("seq2seq_device")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def rewrite(self, text: str) -> str:
        max_len = int(self.config.get("seq2seq_max_gen_len", 128))
        temperature = float(self.config.get("seq2seq_temperature", 0.8))
        top_k = int(self.config.get("seq2seq_top_k", 50))
        # No <hl> tags - full sentence replacement
        source_for_len = f"{self.source_prefix}{text}{self.source_suffix}"
        source_tokens = self.enc.encode(source_for_len)
        capped_len = max(self.min_gen_len, int(len(source_tokens) * self.len_ratio))
        max_len = min(max_len, capped_len)
        return generate_replacement(
            text,
            self.model,
            self.enc,
            max_len,
            temperature,
            top_k,
            self.device,
            self.source_prefix,
            self.source_suffix,
            self.repetition_penalty,
        )
