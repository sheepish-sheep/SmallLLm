"""Quick test script to evaluate seq2seq paraphrasing models."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import (
    load_seq2seq_model,
    load_checkpoint,
    generate_replacement,
)
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding
import json
import torch


def test_model(checkpoint_path: Path, config: dict, test_inputs: list[str], name: str,
                temperature: float = None, top_k: int = None):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"{'='*60}")

    enc = build_hl_encoding()
    model = load_seq2seq_model(config, repo_root)
    load_checkpoint(model, checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    source_prefix = config.get("seq2seq_source_prefix", "paraphrase: ")
    source_suffix = config.get("seq2seq_source_suffix", "")
    # Use provided params or fall back to config
    if temperature is None:
        temperature = float(config.get("seq2seq_temperature", 0.8))
    if top_k is None:
        top_k = int(config.get("seq2seq_top_k", 50))
    max_len = int(config.get("seq2seq_max_gen_len", 80))

    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("-" * 60)

    for i, text in enumerate(test_inputs, 1):
        output = generate_replacement(
            text, model, enc, max_len, temperature, top_k, device,
            source_prefix, source_suffix
        )
        print(f"\n[{i}] Input:  {text}")
        print(f"    Output: {output}")


def test_with_rewriter(config_path: Path, test_inputs: list[str]):
    """Test using Seq2SeqRewriter class with length ratio capping."""
    from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import Seq2SeqRewriter

    print("\n" + "="*70)
    print("TESTING WITH Seq2SeqRewriter (length ratio capping)")
    print("="*70)

    rewriter = Seq2SeqRewriter(config_path)
    print(f"Device: {rewriter.device}")
    print(f"Len ratio: {rewriter.len_ratio}, Min gen len: {rewriter.min_gen_len}")
    print("-" * 60)

    for i, text in enumerate(test_inputs, 1):
        output = rewriter.rewrite(text)
        print(f"\n[{i}] Input:  {text}")
        print(f"    Output: {output}")


def main():
    config_path = repo_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    seq2seq_dir = repo_root / "EncoderAndMoreInput/VN_Pipeline/out/seq2seq"

    # Test inputs - mix of formal sentences to paraphrase into casual VN style
    test_inputs = [
        "I'm not entirely convinced of your claim.",
        "The fellow seems to be loitering about when we're facing a pressing situation.",
        "I haven't encountered this information before.",
        "Would you be so kind as to explain what occurred?",
        "It appears that we have arrived at our destination.",
    ]

    # First test with Seq2SeqRewriter (uses length ratio)
    test_with_rewriter(config_path, test_inputs)

    # Test best checkpoint with greedy decoding and short max_len
    best_path = seq2seq_dir / "best_checkpoint.pt"
    if best_path.exists():
        print("\n" + "="*70)
        print("TESTING BEST CHECKPOINT - GREEDY (temp=0.1, top_k=1, max_len=25)")
        print("="*70)
        config_greedy = config.copy()
        config_greedy["seq2seq_max_gen_len"] = 25
        test_model(best_path, config_greedy, test_inputs, "Best Checkpoint (Greedy)",
                   temperature=0.1, top_k=1)
    else:
        print(f"Best checkpoint not found: {best_path}")


if __name__ == "__main__":
    main()
