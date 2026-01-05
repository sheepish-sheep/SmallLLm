"""
Stub helpers for building a copy/replace dataset.

Goal: build pairs like
  source: "Narrator: The door creaked <hl>REPLACE_ME</hl> in the dark."
  target: "Narrator: The door creaked open in the dark."

Note: `<hl>` and `</hl>` are placeholders. In training, treat them as
special tokens in your tokenizer, not literal text.
"""

import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple


def load_vn_lines(path: str) -> List[str]:
    """
    TODO: Read VN dialogue lines from a cleaned text file.
    - Input path should come from config.json (vn_cleaned_path).
    - Use utf-8 and strip whitespace.
    - Optionally drop very short lines (min length).
    - Return a list of strings (one line per item).
    """
    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parents[3] / path
    lines = []
    if not input_path.is_file():
        raise FileNotFoundError("VN cleaned text file not found")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_copy_replace_pairs(
    lines: Iterable[str],
    highlight_start: str,
    highlight_end: str,
    *,
    span_len: int = 5,
    min_line_len: int = 20,
    max_line_len: int = 160,
    source_prefix: str = "",
    source_suffix: str = "",
    randomize_position: bool = True,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Create (source, target) pairs for seq2seq training.

    For each line, pick a span to wrap with highlight tags.
    - Source: the line with a span wrapped in <hl>...</hl>
    - Target: the original line (model learns to "restore" the highlighted span)

    Args:
        randomize_position: If True, pick random span positions for variety.
                           If False, always use middle (easier to debug).
        seed: Random seed for reproducible results when randomize_position=True.

    Notes:
    - Skips lines outside [min_line_len, max_line_len] range.
    - Avoids inserting tags inside existing markup (lines with < or > are skipped).
    - Word-boundary aware: tries to align spans to word boundaries when possible.
    """
    if randomize_position:
        random.seed(seed)

    pairs = []
    for line in lines:
        if len(line) < min_line_len or len(line) > max_line_len:
            continue
        if len(line) < span_len:
            continue
        # Skip lines that already contain markup
        if "<" in line or ">" in line:
            continue

        max_start = len(line) - span_len
        if randomize_position:
            start = random.randint(0, max_start)
        else:
            # Deterministic: use middle position
            start = max(0, (len(line) - span_len) // 2)

        end = start + span_len

        # Try to align to word boundaries (expand to include full words)
        # Find word start (go back to space or start of line)
        while start > 0 and line[start - 1] not in " \t":
            start -= 1
        # Find word end (go forward to space or end of line)
        while end < len(line) and line[end] not in " \t":
            end += 1

        # Skip if span is now too short or too long
        actual_span_len = end - start
        if actual_span_len < 2 or actual_span_len > span_len * 3:
            # Fall back to original positions
            if randomize_position:
                start = random.randint(0, max_start)
            else:
                start = max(0, (len(line) - span_len) // 2)
            end = start + span_len

        source = line[:start] + highlight_start + line[start:end] + highlight_end + line[end:]
        source = f"{source_prefix}{source}{source_suffix}"
        pairs.append((source, line))

    return pairs


def save_pairs_jsonl(pairs: Iterable[Tuple[str, str]], path: str) -> None:
    """
    Save (source, target) pairs to JSONL.

    Example line:
        {"source": "...", "target": "..."}

    Notes:
    - Ensure the parent directory exists.
    - Use utf-8 and newline="\n".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for source, target in pairs:
            f.write(json.dumps({"source": source, "target": target}) + "\n")


def main() -> None:
    """
    TODO:
    - Read config.json from repo root.
    - Pull highlight_start_token / highlight_end_token.
    - Load VN lines from vn_cleaned_path.
    - Build copy/replace pairs.
    - Split train/val and save JSONL (train.jsonl / val.jsonl).
    """
    config_path = Path(__file__).resolve().parents[3] / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Config file not found")
    with open(config_path, "r") as f:
        config = json.load(f)
    highlight_start = config["highlight_start_token"]
    highlight_end = config["highlight_end_token"]
    lines = load_vn_lines(config["vn_cleaned_path"])
    span_len = int(config.get("seq2seq_span_len", 5))
    min_line_len = int(config.get("seq2seq_min_line_len", 20))
    max_line_len = int(config.get("seq2seq_max_line_len", 160))
    source_prefix = config.get("seq2seq_source_prefix", "")
    source_suffix = config.get("seq2seq_source_suffix", "")
    pairs = build_copy_replace_pairs(
        lines,
        highlight_start,
        highlight_end,
        span_len=span_len,
        min_line_len=min_line_len,
        max_line_len=max_line_len,
        source_prefix=source_prefix,
        source_suffix=source_suffix,
    )
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    output_dir = Path(config.get("pairs_output_dir", config.get("seq2seq_output_dir", "EncoderAndMoreInput/VN_Pipeline/out/seq2seq")))
    save_pairs_jsonl(train_pairs, output_dir / "train.jsonl")
    save_pairs_jsonl(val_pairs, output_dir / "val.jsonl")


if __name__ == "__main__":
    main()
