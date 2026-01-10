"""
Generate paraphrase training data using Claude API.

Reads VN lines, generates paraphrases, saves as training pairs.
"""

import json
import os
import random
import time
from pathlib import Path
import anthropic

# Load API key from file
def load_api_key():
    key_file = Path(__file__).resolve().parents[3] / "api_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ.get("ANTHROPIC_API_KEY")

def load_vn_lines(path: str, min_len: int = 20, max_len: int = 150) -> list[str]:
    """Load and filter VN dialogue lines."""
    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parents[3] / path
    
    lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) >= min_len and len(line) <= max_len:
                # Skip lines with markup
                if "<" not in line and ">" not in line:
                    lines.append(line)
    return lines


def generate_paraphrase(client: anthropic.Anthropic, text: str, max_retries: int = 3) -> str | None:
    """Generate a paraphrase using Claude with retry on rate limit."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Cheapest, fast
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"""Paraphrase this dialogue line naturally. Keep the same meaning but use different words/structure. Output ONLY the paraphrase, nothing else.

Original: {text}

Paraphrase:"""
                }]
            )
            return response.content[0].text.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                # Rate limited - wait and retry
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                print(f"Rate limited, waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"API error: {e}")
                return None
    return None


def main():
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load VN lines
    vn_path = config["vn_cleaned_path"]
    lines = load_vn_lines(vn_path)
    print(f"Loaded {len(lines)} VN lines")
    
    # Shuffle and limit for cost control
    random.seed(42)
    random.shuffle(lines)
    
    # How many to process? Start small to test
    max_lines = int(os.environ.get("MAX_LINES", 5000))
    lines = lines[:max_lines]
    print(f"Processing {len(lines)} lines (set MAX_LINES env var to change)")
    
    # Initialize Claude client
    api_key = load_api_key()
    if not api_key:
        raise ValueError("No API key found! Put it in api_key.txt or set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate paraphrases - write incrementally
    source_prefix = config.get("seq2seq_source_prefix", "SOURCE: ")
    source_suffix = config.get("seq2seq_source_suffix", " => TARGET: ")
    hl_start = config.get("highlight_start_token", "<hl>")
    hl_end = config.get("highlight_end_token", "</hl>")
    
    output_dir = Path(config.get("seq2seq_output_dir", "EncoderAndMoreInput/VN_Pipeline/out/seq2seq"))
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write incrementally to temp file (resume if exists)
    temp_file = output_dir / "pairs_temp.jsonl"
    
    # Check for existing progress
    existing_sources = set()
    success_count = 0
    if temp_file.exists():
        with open(temp_file, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                existing_sources.add(pair["source"])
                success_count += 1
        print(f"Resuming from {success_count} existing pairs", flush=True)
    
    with open(temp_file, "a", encoding="utf-8") as f:  # Append mode
        for i, line in enumerate(lines):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(lines)} ({success_count} pairs)", flush=True)
            
            # Skip if already processed
            source = f"{source_prefix}{hl_start}{line}{hl_end}{source_suffix}"
            if source in existing_sources:
                continue
            
            paraphrase = generate_paraphrase(client, line)
            if paraphrase and len(paraphrase) > 10:
                pair = {"source": source, "target": paraphrase}
                f.write(json.dumps(pair) + "\n")
                f.flush()  # Write immediately
                success_count += 1
            
            time.sleep(1.3)  # 50 req/min limit = 1 req per 1.2s, add buffer
    
    print(f"Generated {success_count} paraphrase pairs", flush=True)
    
    # Read back and split train/val
    pairs = []
    with open(temp_file, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    
    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")
    
    # Clean up temp
    temp_file.unlink()
    
    print(f"Saved {len(train_pairs)} train, {len(val_pairs)} val pairs", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
