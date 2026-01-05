# VN Pipeline

This folder contains scripts for fine-tuning on VN text, training a copy/replace 
encoder-decoder, and calling a voice model for audio synthesis.

## Folder Structure

```
VN_Pipeline/
├── pipeline_stub.py          # Main entry point - runs all stages
├── voice_stub.py             # Voice synthesis integration
├── README.md
├── config.example.json       # Template config (copy to repo root as config.json)
│
├── training/                 # Training-related code
│   ├── seq2seq_train_stub.py     # Encoder-decoder training
│   ├── finetune_vn_stub.py       # Fine-tune base model on VN text
│   ├── data_prep_stub.py         # Create copy/replace pairs
│   └── vn_shard_builder_stub.py  # Build token shards from VN text
│
├── inference/                # Generation/inference
│   ├── seq2seq_infer.py          # Seq2SeqRewriter class for inference
│   ├── generate_base.py          # Generate with base checkpoint
│   └── generate_vn.py            # Generate with fine-tuned model
│
├── eval/                     # Evaluation and visualization
│   ├── seq2seq_eval_stub.py      # Evaluate model metrics
│   └── plot_seq2seq_metrics_stub.py  # Plot training curves
│
├── utils/                    # Shared utilities
│   ├── specialtoken_hl.py        # Highlight token encoding (implemented)
│   ├── config_schema.py          # Config validation
│   └── latest_checkpoint_stub.py # Find newest checkpoint
│
├── examples/                 # Sample inputs/outputs
│   ├── expected_output.txt
│   └── highlighted_input.txt
│
└── out/                      # Generated outputs
    ├── finetune/             # Fine-tuning checkpoints
    ├── finetune_runs/        # Fine-tuning run history
    └── seq2seq/              # Seq2seq checkpoints and data
```

## Quick Start

### 1. Set up config
```bash
# Copy example config to repo root
cp EncoderAndMoreInput/VN_Pipeline/config.example.json config.json
```

Edit `config.json` and set:
- `vn_cleaned_path`: Path to your cleaned VN dialogue text
- `base_checkpoint_path`: Path to base model checkpoint (e.g., `log/model_04999.pt`)
- `vn_shard_dir`: Where to save tokenized shards

### 2. Run individual stages
```bash
# From repo root, with venv activated
python -m EncoderAndMoreInput.VN_Pipeline.training.data_prep_stub
python -m EncoderAndMoreInput.VN_Pipeline.training.finetune_vn_stub
python -m EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub
```

### 3. Or run the full pipeline
```bash
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub

# Run specific stages only
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages data_prep seq2seq
```

### 4. Generate text
```bash
# With base model
python -m EncoderAndMoreInput.VN_Pipeline.inference.generate_base --prompt "Hello"

# With fine-tuned model
python -m EncoderAndMoreInput.VN_Pipeline.inference.generate_vn --prompt "Hello"
```

## Implementation Order

If implementing from scratch, follow this order:

1. `utils/specialtoken_hl.py` ✓ (already implemented)
2. `utils/latest_checkpoint_stub.py`
3. `training/vn_shard_builder_stub.py`
4. `training/finetune_vn_stub.py`
5. `training/data_prep_stub.py`
6. `training/seq2seq_train_stub.py`
7. `inference/seq2seq_infer.py`
8. `eval/seq2seq_eval_stub.py`
9. `eval/plot_seq2seq_metrics_stub.py`
10. `voice_stub.py`
11. `pipeline_stub.py`

## Pipeline Flow

```
┌─────────────────┐
│  VN Text Data   │
│ (cleaned .txt)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ vn_shard_builder│────▶│  Token Shards   │
│                 │     │ (train/val.npy) │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  finetune_vn    │────▶│ Fine-tuned GPT  │
│                 │     │  checkpoint     │
└─────────────────┘     └────────┬────────┘
                                 │
         ┌───────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   data_prep     │────▶│  JSONL Pairs    │
│ (highlight tags)│     │ (source/target) │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  seq2seq_train  │────▶│ Encoder-Decoder │
│                 │     │   checkpoint    │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  seq2seq_infer  │────▶│  Rewritten Text │
│                 │     │ (highlights     │
└─────────────────┘     │  replaced)      │
                        └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│   voice_stub    │────▶│   Audio Files   │
│                 │     │     (.wav)      │
└─────────────────┘     └─────────────────┘
```

## Config Keys

See `utils/config_schema.py` for full documentation. Key settings:

| Key | Description |
|-----|-------------|
| `vn_cleaned_path` | Path to cleaned VN dialogue text |
| `base_checkpoint_path` | Base GPT checkpoint to fine-tune from |
| `vn_shard_dir` | Directory for tokenized shards |
| `finetune_output_dir` | Where to save fine-tuned checkpoints |
| `seq2seq_output_dir` | Where to save seq2seq checkpoints and data |
| `highlight_start_token` | Start tag for highlights (default: `<hl>`) |
| `highlight_end_token` | End tag for highlights (default: `</hl>`) |

## Importing Modules

After reorganization, use these import paths:

```python
# Training
from EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub import ...
from EncoderAndMoreInput.VN_Pipeline.training.data_prep_stub import ...

# Inference
from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import Seq2SeqRewriter

# Utils
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding
from EncoderAndMoreInput.VN_Pipeline.utils.config_schema import validate_config

# Eval
from EncoderAndMoreInput.VN_Pipeline.eval.seq2seq_eval_stub import ...
```

## Documentation

Detailed documentation with diagrams:

| Document | Description |
|----------|-------------|
| [ENCODER_DECODER_ARCHITECTURE.md](../../docs/EncoderAndMoreInput/VN_Pipeline/ENCODER_DECODER_ARCHITECTURE.md) | Complete encoder-decoder architecture with layer-by-layer diagrams |
| [TRAINING_METRICS_GRAPHS.md](../../docs/EncoderAndMoreInput/VN_Pipeline/TRAINING_METRICS_GRAPHS.md) | Metrics tracking, graph interpretation, troubleshooting |
| [VOICE_INTEGRATION.md](../../docs/EncoderAndMoreInput/VN_Pipeline/VOICE_INTEGRATION.md) | Voice cloning module, configuration, usage |
| [HIGHLIGHT_TOKENIZATION.md](../../docs/EncoderAndMoreInput/VN_Pipeline/HIGHLIGHT_TOKENIZATION.md) | How `<hl>` tokens work |
| [CONFIG_REFERENCE.md](../../docs/EncoderAndMoreInput/VN_Pipeline/CONFIG_REFERENCE.md) | All config.json settings |
| [FOLDER_STRUCTURE.md](../../docs/EncoderAndMoreInput/VN_Pipeline/FOLDER_STRUCTURE.md) | Project file organization |

## References

- `EncoderAndMoreInput/encoder_decoder_backup.py` - Encoder-decoder model architecture
- `docs/EncoderAndMoreInput/TrainingLoop/*.md` - Training loop explanations
- `Basic AI Voice Clone/` - Voice synthesis module
