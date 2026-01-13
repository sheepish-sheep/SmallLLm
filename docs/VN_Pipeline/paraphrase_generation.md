# Paraphrase Generation with Claude API

This document explains how `generate_paraphrases.py` works to create training data for the seq2seq model.

## Overview

The script uses Claude (Anthropic's AI) to generate paraphrases of visual novel dialogue lines. These paraphrases become training data so our seq2seq model learns to rewrite text naturally.

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: VN Dialogue File                                        │
│  "Then how's this for comprehension?"                           │
│  "Wilhelm was probably far more dangerous than those three."    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROCESS: Claude API (for each line)                            │
│                                                                 │
│  Prompt: "Paraphrase this dialogue line naturally..."           │
│  Response: "So, how do you understand that?"                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Training Pairs (JSONL)                                 │
│                                                                 │
│  source: "SOURCE: <hl>Then how's this for comprehension?</hl>   │
│           => TARGET: "                                          │
│  target: "So, how do you understand that?"                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Walkthrough

### 1. Load API Key (`load_api_key`)

```python
def load_api_key():
    key_file = Path(__file__).resolve().parents[3] / "api_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ.get("ANTHROPIC_API_KEY")
```

**What it does:**
- Looks for `api_key.txt` in the project root
- Falls back to `ANTHROPIC_API_KEY` environment variable
- Returns the API key string

**Why:**
- Keeps sensitive API key out of code
- Flexible: works with file or env var

---

### 2. Load VN Lines (`load_vn_lines`)

```python
def load_vn_lines(path: str, min_len: int = 20, max_len: int = 150) -> list[str]:
    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parents[3] / path
    
    lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) >= min_len and len(line) <= max_len:
                if "<" not in line and ">" not in line:
                    lines.append(line)
    return lines
```

**What it does:**
1. Resolves the file path (relative or absolute)
2. Reads each line from the VN text file
3. Filters lines by length (20-150 characters)
4. Skips lines with `<` or `>` (markup)

**Why filter?**
- Too short: Not enough context for meaningful paraphrase
- Too long: More expensive API calls, harder to paraphrase
- Markup: Would confuse the model

---

### 3. Generate Paraphrase (`generate_paraphrase`)

```python
def generate_paraphrase(client: anthropic.Anthropic, text: str) -> str | None:
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
        print(f"API error: {e}")
        return None
```

**What it does:**
1. Sends a request to Claude API
2. Uses `claude-3-haiku` model (cheapest, ~$0.25/million tokens)
3. Asks Claude to paraphrase the text naturally
4. Returns the paraphrase or `None` on error

**The prompt breakdown:**
```
"Paraphrase this dialogue line naturally."     ← Main instruction
"Keep the same meaning"                         ← Preserve semantics
"but use different words/structure."            ← Force variation
"Output ONLY the paraphrase, nothing else."     ← No explanations
```

**API Response Structure:**
```python
response.content[0].text  # The actual text Claude generated
```

---

### 4. Main Processing Loop

```python
with open(temp_file, "w", encoding="utf-8") as f:
    for i, line in enumerate(lines):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(lines)} ({success_count} pairs)", flush=True)
        
        paraphrase = generate_paraphrase(client, line)
        if paraphrase and len(paraphrase) > 10:
            source = f"{source_prefix}{hl_start}{line}{hl_end}{source_suffix}"
            pair = {"source": source, "target": paraphrase}
            f.write(json.dumps(pair) + "\n")
            f.flush()  # Write immediately
            success_count += 1
        
        time.sleep(0.05)  # Rate limit protection
```

**Step by step:**

| Step | Code | Purpose |
|------|------|---------|
| 1 | `for i, line in enumerate(lines)` | Loop through all VN lines |
| 2 | `if i % 50 == 0: print(...)` | Show progress every 50 lines |
| 3 | `generate_paraphrase(client, line)` | Call Claude API |
| 4 | `if paraphrase and len(...) > 10` | Skip failed/empty responses |
| 5 | `source = f"{prefix}{hl_start}{line}{hl_end}{suffix}"` | Format source with highlights |
| 6 | `f.write(json.dumps(pair) + "\n")` | Save to temp file |
| 7 | `f.flush()` | Write immediately (see progress) |
| 8 | `time.sleep(0.05)` | Small delay to avoid rate limits |

**Why write incrementally?**
- Can monitor progress in real-time
- If script crashes, don't lose all progress
- Can Ctrl+C and still have partial data

---

### 5. Train/Val Split

```python
random.shuffle(pairs)
split_idx = int(len(pairs) * 0.95)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]
```

**What it does:**
1. Shuffles all pairs randomly
2. Takes 95% for training
3. Takes 5% for validation

**Why 95/5?**
- Training needs more data
- Validation just needs enough to measure loss
- Standard split ratio for NLP

---

### 6. Output Format

Each line in the JSONL file:

```json
{
  "source": "SOURCE: <hl>Then how's this for comprehension?</hl> => TARGET: ",
  "target": "So, how do you understand that?"
}
```

**Why this format?**
- `<hl>...</hl>` tells the model what to replace
- Entire sentence is highlighted = replace everything
- `SOURCE: ... => TARGET:` is the prompt format the seq2seq expects

---

## Usage

### Basic Run (5,000 lines, ~$1)
```powershell
python -m EncoderAndMoreInput.VN_Pipeline.training.generate_paraphrases
```

### Full Dataset (~56,000 lines, ~$10-15)
```powershell
$env:MAX_LINES = "56000"
python -m EncoderAndMoreInput.VN_Pipeline.training.generate_paraphrases
```

### Custom Amount
```powershell
$env:MAX_LINES = "10000"
python -m EncoderAndMoreInput.VN_Pipeline.training.generate_paraphrases
```

---

## Cost Estimation

| Lines | Claude Haiku Cost | Time |
|-------|-------------------|------|
| 1,000 | ~$0.20 | ~2 min |
| 5,000 | ~$1.00 | ~10 min |
| 10,000 | ~$2.00 | ~20 min |
| 56,000 | ~$10-15 | ~90 min |

---

## Files

| File | Purpose |
|------|---------|
| `api_key.txt` | Your Anthropic API key |
| `config.json` | Paths and settings |
| `pairs_temp.jsonl` | Temporary file during generation |
| `train.jsonl` | Final training data (95%) |
| `val.jsonl` | Final validation data (5%) |

---

## Troubleshooting

### "No API key found"
- Create `api_key.txt` in project root with your key
- Or set `ANTHROPIC_API_KEY` environment variable

### "API error: rate limit"
- Increase `time.sleep()` value
- Claude Haiku has generous limits, usually not an issue

### Script crashes mid-way
- Check `pairs_temp.jsonl` for partial progress
- You can manually split it into train/val
