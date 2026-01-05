"""
Stub for evaluating Seq2SeqRewriter quality.

Goals:
- Track output length ratio vs input length.
- Track copy match rate outside <hl>...</hl>.
- Track exact match rate (optional).
- Append metrics to a CSV for plotting.

CSV columns: step, timestamp, val_loss, avg_len_ratio, copy_match_rate, exact_match_rate
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import Seq2SeqRewriter
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding


def load_eval_pairs(config_path: Path, max_pairs: int = 100) -> list[dict]:
    """
    TODO: Load a fixed eval set from JSONL.
    
    Steps:
    1. Open config_path (use the parameter, don't redefine it).
    2. Get "eval_path" from config, or fall back to seq2seq_output_dir/val.jsonl.
    3. Convert string paths to Path objects before calling .is_file().
    4. Load JSONL lines and parse each as JSON.
    5. Return first N pairs for consistent evaluation.
    
    Expected JSONL format: {"source": "...", "target": "..."}
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    eval_path = Path(config.get("eval_path", Path(config.get("seq2seq_output_dir", "EncoderAndMoreInput/VN_Pipeline/out/seq2seq")) / "val.jsonl"))
    if not eval_path.is_file():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")
    with open(eval_path, "r") as f:
        pairs = [json.loads(line) for line in f]
    return pairs[:max_pairs]


def compute_length_ratio(outputs: Iterable[str], inputs: Iterable[str]) -> float:
    """
    TODO: Compute average length ratio.
    
    Steps:
    1. Convert iterables to lists (they can only be iterated once).
    2. Choose length method: character len() or token count (be consistent with training).
    3. For each pair: ratio = len(output) / len(input), handle div-by-zero.
    4. Return mean of all ratios.
    
    IMPORTANT: Ideal ratio depends on your task. For highlight rewriting:
    - Ratio ~1.0 means similar length (good for subtle rewrites).
    - Ratio >> 1 means output is much longer (may indicate repetition bug).
    - Ratio << 1 means output is truncated (may indicate early stopping).
    """
    it_outputs = list(outputs)
    it_inputs = list(inputs)
    if len(it_inputs) == 0:
        return 0.0
    if len(it_outputs) == 0:
        return 0.0
    lengths = []
    for output, input in zip(it_outputs, it_inputs):
        if len(input) == 0 or len(output) == 0:
            continue
        lengths.append(len(output) / len(input))
    if len(lengths) == 0:
        return 0.0
    return sum(lengths) / len(lengths)


def compute_copy_match_rate(
    outputs: Iterable[str],
    sources: Iterable[str],
    highlight_start: str,
    highlight_end: str,
) -> float:
    """
    TODO: Compute copy match rate for non-highlighted portions.
    
    PURPOSE: Verify the model preserves text OUTSIDE the <hl>...</hl> span.
    
    Steps:
    1. Convert iterables to lists.
    2. For each (output, source) pair:
       a. Extract portions of source BEFORE and AFTER the highlight span.
       b. Check if those portions appear unchanged in output.
       c. Score: binary (all match or not) or fuzzy (% chars matching).
    3. Return mean score across all pairs.
    
    Example:
      source: "The <hl>quick</hl> brown fox"
      output: "The fast brown fox"
      -> "The " and " brown fox" should match exactly in output.
    """
    it_outputs = list(outputs)
    it_sources = list(sources)
    if len(it_outputs) == 0:
        return 0.0
    match_count = 0
    for output, source in zip(it_outputs, it_sources):
        hl_start_idx = source.find(highlight_start)
        hl_end_idx = source.find(highlight_end)
        if hl_start_idx == -1 or hl_end_idx == -1:
            continue  # Skip pairs without markers
        source_before = source[:hl_start_idx]
        source_after = source[hl_end_idx + len(highlight_end):]
        # Output has no markers, so check if it preserves before/after text
        if output.startswith(source_before) and output.endswith(source_after):
            match_count += 1
    return match_count / len(it_outputs)


def compute_exact_match_rate(outputs: Iterable[str], targets: Iterable[str]) -> float:
    """
    TODO: Compute exact match rate.
    
    Steps:
    1. Convert iterables to lists.
    2. Count pairs where output.strip() == target.strip().
    3. Return count / total.
    """
    it_outputs = list(outputs)
    it_targets = list(targets)
    match_count = 0
    for output, target in zip(it_outputs, it_targets):
        if output.strip() == target.strip():
            match_count += 1
    if not it_outputs:
        return 0.0
    return match_count / len(it_outputs)
    


def evaluate_rewriter(config_path: Path, output_csv: Path, step: int = 0) -> None:
    """
    TODO: Run Seq2SeqRewriter on eval pairs and log metrics.
    
    Steps:
    1. Load config from config_path.
    2. Load eval pairs via load_eval_pairs().
    3. Instantiate Seq2SeqRewriter (from seq2seq_infer.py).
    4. For each source, call rewriter.rewrite(source) to get output.
    5. Compute all metrics.
    6. Append row to CSV with headers if new file.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)
    
    output_csv = Path(output_csv)
    if not output_csv.is_file():
        with open(output_csv, "w") as f:
            f.write("step,timestamp,val_loss,avg_len_ratio,copy_match_rate,exact_match_rate\n")
    
    rewriter = Seq2SeqRewriter(config)
    all_outputs = []
    all_sources = []
    all_targets = []
    for pair in load_eval_pairs(config_path):
        output = rewriter.rewrite(pair["source"])
        all_outputs.append(output)
        all_sources.append(pair["source"])
        all_targets.append(pair["target"])
    
    avg_len_ratio = compute_length_ratio(all_outputs, all_sources)
    copy_match_rate = compute_copy_match_rate(all_outputs, all_sources, config["highlight_start_token"], config["highlight_end_token"])
    exact_match_rate = compute_exact_match_rate(all_outputs, all_targets)
    
    metrics = {
        "step": step,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "val_loss": None,
        "avg_len_ratio": avg_len_ratio,
        "copy_match_rate": copy_match_rate,
        "exact_match_rate": exact_match_rate,
    }
    with open(output_csv, "a") as f:
        f.write(f"{metrics['step']},{metrics['timestamp']},{metrics['val_loss']},{metrics['avg_len_ratio']},{metrics['copy_match_rate']},{metrics['exact_match_rate']}\n")


def main() -> None:
    """
    TODO: CLI entry point.
    
    Steps:
    1. Parse args: --config (default: repo_root/config.json), --step (optional).
    2. Determine output CSV path from config["seq2seq_output_dir"].
    3. Call evaluate_rewriter(config_path, output_csv, step).
    """
    config_path = Path("EncoderAndMoreInput/VN_Pipeline/config.json")
    output_csv = Path("EncoderAndMoreInput/VN_Pipeline/out/seq2seq/metrics.csv")
    step = 0
    evaluate_rewriter(config_path, output_csv, step)
    


if __name__ == "__main__":
    main()
