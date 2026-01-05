# VN Pipeline Folder Structure

This document explains the folder organization and how to import from each module.

## Directory Layout

```
EncoderAndMoreInput/VN_Pipeline/
│
├── pipeline_stub.py          # Main entry point
├── voice_stub.py             # Voice synthesis (at root for easy access)
├── README.md                 # Overview and quick start
├── config.example.json       # Template config
│
├── training/                 # Training scripts
│   ├── __init__.py
│   ├── seq2seq_train_stub.py     # Encoder-decoder training
│   ├── finetune_vn_stub.py       # Fine-tune GPT on VN text
│   ├── data_prep_stub.py         # Build training pairs
│   └── vn_shard_builder_stub.py  # Tokenize VN text to shards
│
├── inference/                # Generation scripts
│   ├── __init__.py
│   ├── seq2seq_infer.py          # Seq2SeqRewriter class
│   ├── generate_base.py          # CLI for base model
│   └── generate_vn.py            # CLI for fine-tuned model
│
├── eval/                     # Evaluation
│   ├── __init__.py
│   ├── seq2seq_eval_stub.py      # Compute metrics
│   └── plot_seq2seq_metrics_stub.py  # Plot graphs
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── specialtoken_hl.py        # Highlight tokenizer
│   ├── config_schema.py          # Config validation
│   └── latest_checkpoint_stub.py # Find checkpoints
│
├── examples/                 # Sample data
│   ├── expected_output.txt
│   └── highlighted_input.txt
│
└── out/                      # Generated files
    ├── finetune/
    ├── finetune_runs/
    └── seq2seq/
```

## Import Paths

### From training/

```python
# Seq2seq training
from EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub import (
    load_seq2seq_model,
    build_seq2seq_dataloader,
    train_seq2seq_loop,
)

# Fine-tuning
from EncoderAndMoreInput.VN_Pipeline.training.finetune_vn_stub import (
    load_base_checkpoint,
    train_finetune_loop,
)

# Data preparation
from EncoderAndMoreInput.VN_Pipeline.training.data_prep_stub import (
    build_copy_replace_pairs,
    save_pairs_jsonl,
)
```

### From inference/

```python
# Seq2seq rewriter
from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import (
    Seq2SeqRewriter,
    generate_replacement,
)
```

### From utils/

```python
# Tokenization
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import (
    build_hl_encoding,
    HL_START,
    HL_END,
)

# Config validation
from EncoderAndMoreInput.VN_Pipeline.utils.config_schema import (
    validate_config,
    CONFIG_SCHEMA,
)
```

### From eval/

```python
# Evaluation
from EncoderAndMoreInput.VN_Pipeline.eval.seq2seq_eval_stub import (
    load_eval_pairs,
    compute_length_ratio,
    evaluate_rewriter,
)
```

## Running Scripts

### As Python Modules (Recommended)

```bash
# From repo root
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub
python -m EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub
python -m EncoderAndMoreInput.VN_Pipeline.inference.generate_vn --prompt "Hello"
```

### Direct Execution

```bash
# From VN_Pipeline directory
cd EncoderAndMoreInput/VN_Pipeline
python pipeline_stub.py
python training/seq2seq_train_stub.py
```

## Path Resolution

Files in subfolders use `parents[3]` to find the repo root:

```python
# In training/seq2seq_train_stub.py
repo_root = Path(__file__).resolve().parents[3]
#           training/ → VN_Pipeline/ → EncoderAndMoreInput/ → Chuni/
#           [0]         [1]            [2]                    [3]
```

Files at the VN_Pipeline root use `parents[2]`:

```python
# In pipeline_stub.py
repo_root = Path(__file__).resolve().parents[2]
#           VN_Pipeline/ → EncoderAndMoreInput/ → Chuni/
#           [0]            [1]                    [2]
```

## Why This Structure?

| Folder | Purpose | Files |
|--------|---------|-------|
| `training/` | All training logic | 4 files |
| `inference/` | Generation/serving | 3 files |
| `eval/` | Quality metrics | 2 files |
| `utils/` | Shared helpers | 3 files |
| Root | Entry points | 2 files |

Benefits:
- **Clear separation**: Training vs inference vs eval
- **Easier navigation**: Find files by purpose
- **Scalable**: Add more files to each folder without clutter
- **Standard layout**: Matches common Python project structure

