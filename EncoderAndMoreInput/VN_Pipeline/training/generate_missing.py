"""
Efficiently generate only MISSING paraphrase variants.
Much faster than scanning all lines.
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict
import anthropic

def load_api_key():
    key_file = Path(__file__).resolve().parents[3] / "api_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ.get("ANTHROPIC_API_KEY")

def generate_paraphrase(client, text: str, variant: int, max_retries: int = 3) -> str | None:
    style_hints = [
        "Use casual, conversational language.",
        "Use slightly different sentence structure.",
        "Rephrase with synonyms while keeping the tone.",
    ]
    style_hint = style_hints[variant % len(style_hints)]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"""Paraphrase this dialogue line for a visual novel. {style_hint}

Rules:
- Keep the SAME length (within 10% of original)
- Keep the same meaning and emotional tone
- Use different words and structure
- Output ONLY the paraphrase, no quotes or labels

Original: {text}

Paraphrase:"""
                }]
            )
            result = response.content[0].text.strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result
        except Exception as e:
            err_str = str(e).lower()
            # Retry on rate limits, overload, or connection errors
            if "429" in err_str or "rate_limit" in err_str or "overload" in err_str or "connection" in err_str or "529" in err_str:
                wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}", flush=True)
                print(f"Retrying in {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"API error (not retrying): {e}")
                return None
    return None

def main():
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    output_dir = repo_root / config.get("seq2seq_output_dir", "EncoderAndMoreInput/VN_Pipeline/out/seq2seq")
    temp_file = output_dir / "pairs_temp.jsonl"

    source_prefix = config.get("seq2seq_source_prefix", "paraphrase: ")
    hl_start = config.get("highlight_start_token", "<hl>")
    hl_end = config.get("highlight_end_token", "</hl>")

    # Load existing pairs and find what's missing
    print("Analyzing existing data...", flush=True)
    variant_counts = defaultdict(set)
    with open(temp_file, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            source = pair['source']
            variant = pair.get('variant', 0)
            start = source.find(hl_start) + len(hl_start)
            end = source.find(hl_end)
            if start > len(hl_start) - 1 and end > start:
                original = source[start:end]
                variant_counts[original].add(variant)

    # Find missing variants
    missing = []
    for original, variants in variant_counts.items():
        for v in range(4):
            if v not in variants:
                missing.append((original, v))

    print(f"Found {len(missing)} missing variants to generate", flush=True)

    if not missing:
        print("Nothing to generate!")
        return

    # Initialize Claude client
    api_key = load_api_key()
    if not api_key:
        raise ValueError("No API key found!")
    client = anthropic.Anthropic(api_key=api_key)

    # Generate missing variants
    success_count = 0
    with open(temp_file, "a", encoding="utf-8") as f:
        for i, (original, variant) in enumerate(missing):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(missing)} ({success_count} generated)", flush=True)

            paraphrase = generate_paraphrase(client, original, variant)
            if paraphrase and len(paraphrase) > 10:
                ratio = len(paraphrase) / len(original)
                if 0.7 < ratio < 1.5:
                    source = f"{source_prefix}{hl_start}{original}{hl_end}"
                    pair = {"source": source, "target": paraphrase, "variant": variant}
                    f.write(json.dumps(pair) + "\n")
                    f.flush()
                    success_count += 1

            time.sleep(0.1)

    print(f"Generated {success_count} new pairs", flush=True)
    print(f"Total pairs now: {len(variant_counts) * 4 - len(missing) + success_count}", flush=True)

if __name__ == "__main__":
    main()
