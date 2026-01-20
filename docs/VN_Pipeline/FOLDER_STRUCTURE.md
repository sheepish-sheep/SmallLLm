# VN Pipeline Folder Structure

This document explains the folder organization and how to import from each module.

## Directory Layout

```
EncoderAndMoreInput/VN_Pipeline/
│
├── pipeline_stub.py          # Main entry point - orchestrates all stages
├── voice_stub.py             # Voice synthesis wrapper
│
├── training/                 # Training scripts
│   ├── __init__.py
│   ├── seq2seq_train_stub.py     # Encoder-decoder training (local)
│   ├── seq2seq_train_modal.py    # Training on Modal cloud GPU
│   ├── finetune_vn_stub.py       # Fine-tune GPT on VN text
│   ├── data_prep_stub.py         # Build training pairs from VN dialogue
│   ├── generate_paraphrases.py   # Generate paraphrase pairs using Claude API
│   ├── upload_to_modal.py        # Upload data/config to Modal volume
│   └── vn_shard_builder_stub.py  # Tokenize VN text to shards
│
├── inference/                # Generation scripts
│   ├── __init__.py
│   ├── seq2seq_infer.py          # Seq2SeqRewriter class - core inference
│   ├── rewrite_and_speak.py      # Full pipeline: paraphrase → voice
│   ├── generate_base.py          # CLI for base model generation
│   └── generate_vn.py            # CLI for fine-tuned model generation
│
├── eval/                     # Evaluation
│   ├── __init__.py
│   ├── seq2seq_eval_stub.py      # Compute metrics
│   └── plot_seq2seq_metrics_stub.py  # Plot training graphs
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── specialtoken_hl.py        # Highlight tokenizer (<hl> tokens)
│   ├── config_schema.py          # Config validation and schema
│   └── latest_checkpoint_stub.py # Find latest checkpoint
│
├── docs/                     # Pipeline-specific documentation
│   └── modal_training.md         # Guide for Modal cloud training
│
└── out/                      # Generated outputs
    ├── finetune_runs/            # Fine-tuning checkpoints
    ├── seq2seq/                  # Seq2seq checkpoints, data, metrics
    │   ├── best_checkpoint.pt    # Best model by validation loss
    │   ├── model_final.pt        # Final checkpoint after training
    │   ├── metrics.csv           # Training metrics log
    │   ├── train.jsonl           # Training data
    │   └── val.jsonl             # Validation data
    └── audio/                    # Generated voice files
```

## Key Files Explained

### Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `pipeline_stub.py` | Orchestrates all pipeline stages | `python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub` |
| `inference/rewrite_and_speak.py` | Full inference: text → paraphrase → voice | `python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak -i` |

### Training

| File | Purpose |
|------|---------|
| `seq2seq_train_stub.py` | Local training with EncoderDecoder model |
| `seq2seq_train_modal.py` | Cloud training on Modal (A10G GPU) |
| `generate_paraphrases.py` | Generate training pairs using Claude API |
| `data_prep_stub.py` | Create highlight-based training pairs |
| `upload_to_modal.py` | Upload data and config to Modal volume |

### Inference

| File | Purpose |
|------|---------|
| `seq2seq_infer.py` | Core Seq2SeqRewriter class for generation |
| `rewrite_and_speak.py` | Chains paraphrasing + voice synthesis |

## Import Paths

### Inference (Most Common)

```python
# Full pipeline: rewrite + speak
from EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak import (
    rewrite_and_speak,      # Full pipeline
    rewrite_text,           # Just paraphrasing
    speak_text,             # Just voice
)

# Seq2seq rewriter only
from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import (
    Seq2SeqRewriter,        # High-level class
    generate_replacement,   # Low-level function
    load_seq2seq_model,     # Load model architecture
    load_checkpoint,        # Load weights
)
```

### Training

```python
from EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub import (
    load_seq2seq_model,
    build_seq2seq_dataloader,
    train_seq2seq_loop,
    init_seq2seq_from_gpt,  # Initialize from pretrained
)

from EncoderAndMoreInput.VN_Pipeline.training.generate_paraphrases import (
    generate_paraphrase_pairs,
)
```

### Utilities

```python
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import (
    build_hl_encoding,      # Get tokenizer with <hl> tokens
    HL_START,               # "<hl>" token
    HL_END,                 # "</hl>" token
)

from EncoderAndMoreInput.VN_Pipeline.utils.config_schema import (
    validate_config,
    CONFIG_SCHEMA,
)
```

## Running Scripts

### Recommended: As Python Modules

```bash
# From repo root (C:\Users\...\Chuni)

# Full pipeline orchestration
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages seq2seq voice

# Interactive paraphrase + voice
python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak -i

# Test seq2seq model
python -m EncoderAndMoreInput.VN_Pipeline.inference.test_seq2seq

# Train on Modal
modal run EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_modal.py
```

### Direct Execution

```bash
cd EncoderAndMoreInput/VN_Pipeline
python pipeline_stub.py
python inference/rewrite_and_speak.py -i
```

## Path Resolution

Files use `Path(__file__).resolve().parents[N]` to find the repo root:

```python
# In training/seq2seq_train_stub.py (3 levels deep)
repo_root = Path(__file__).resolve().parents[3]
#           training/ → VN_Pipeline/ → EncoderAndMoreInput/ → Chuni/
#           [0]         [1]            [2]                    [3]

# In pipeline_stub.py (2 levels deep)
repo_root = Path(__file__).resolve().parents[2]
#           VN_Pipeline/ → EncoderAndMoreInput/ → Chuni/
#           [0]            [1]                    [2]
```

## Folder Purpose Summary

| Folder | Purpose | Key Files |
|--------|---------|-----------|
| `training/` | All training logic | seq2seq_train_stub.py, generate_paraphrases.py |
| `inference/` | Generation/serving | seq2seq_infer.py, rewrite_and_speak.py |
| `eval/` | Quality metrics | seq2seq_eval_stub.py |
| `utils/` | Shared helpers | specialtoken_hl.py, config_schema.py |
| `out/` | Generated outputs | checkpoints, data, audio |

## Basic AI Voice Clone Folder

The voice synthesis system lives in `Basic AI Voice Clone/`:

```
Basic AI Voice Clone/
├── main.py                 # Background service with keybind listener
├── voice_cloner.py         # Chatterbox TTS voice cloner
├── voice_cloner_xtts.py    # XTTS-v2 voice cloner (better quality)
├── config.py               # Local config (keybinds, paths)
├── overlay_window.py       # Transparent overlay UI
├── chatterbox/             # Chatterbox TTS model code
├── voice_samples/          # Voice reference audio files
│   └── ma/                 # Japanese voice samples (.ogg files)
└── audio_output/           # Generated audio files
```

### Voice Model Selection

Set `voice_model_type` in `config.json` to choose between models:

| Model | Config Value | Quality | Speed |
|-------|--------------|---------|-------|
| XTTS-v2 | `"xtts"` | Better, more natural | ~0.5x real-time |
| Chatterbox | `"chatterbox"` | Good | ~0.3x real-time |

```json
{
  "voice_model_type": "xtts"  // or "chatterbox"
}
```
