# Modal Training Guide

This guide explains how to train the seq2seq model on Modal for faster iteration (~1-2 sec/step vs ~60-90 sec/step locally).

## Prerequisites

1. **Modal account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI installed**: `pip install modal`
3. **Modal authenticated**: `modal token new`

---

## Volume Setup (One-Time)

Modal uses **volumes** for persistent storage. Create one for checkpoints:

```powershell
modal volume create chuni-checkpoints
```

---

## File Structure on Modal Volume

```
chuni-checkpoints/
├── config.json                    # Training configuration
├── model_04999_clean.pt           # Base GPT checkpoint
├── finetune_runs/
│   └── best_checkpoint.pt         # VN fine-tuned checkpoint
└── seq2seq/
    ├── train.jsonl                # Training data
    ├── val.jsonl                  # Validation data
    ├── model_00250.pt             # Checkpoint to resume from
    ├── best_checkpoint.pt         # Best model (auto-saved)
    └── metrics.csv                # Training metrics
```

---

## Uploading Files to Modal

### 1. Upload Training Data

```powershell
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/train.jsonl" "seq2seq/train.jsonl"
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/val.jsonl" "seq2seq/val.jsonl"
```

### 2. Upload Configuration

```powershell
modal volume put chuni-checkpoints "config.json" "config.json"
```

### 3. Upload Base Model (First Time Only)

```powershell
modal volume put chuni-checkpoints "log/model_04999_clean.pt" "model_04999_clean.pt"
```

### 4. Upload VN Fine-tuned Checkpoint (First Time Only)

```powershell
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs/best_checkpoint.pt" "finetune_runs/best_checkpoint.pt"
```

### 5. Upload Checkpoint to Resume From (If Resuming)

```powershell
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/model_00250.pt" "seq2seq/model_00250.pt"
```

---

## Configuration for Modal

Before uploading `config.json`, ensure these settings:

### Fresh Training (VN Init)
```json
{
  "seq2seq_force_vn_init": true,
  "seq2seq_resume_path": ""
}
```

### Resume from Checkpoint
```json
{
  "seq2seq_force_vn_init": false,
  "seq2seq_resume_path": "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/model_00250.pt"
}
```

---

## Running Training on Modal

```powershell
modal run EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_modal.py
```

This will:
1. Spin up a GPU instance (A10G or similar)
2. Load files from the volume
3. Train the model
4. Save checkpoints back to the volume

---

## Monitoring Progress

Training output appears in your terminal. Look for:

```
Step 250 | Loss: 5.234 | Val Loss: 5.567 | ...
Step 300 | Loss: 4.891 | Val Loss: 5.123 | ...
```

---

## Downloading Results

### Download Best Checkpoint

```powershell
modal volume get chuni-checkpoints "seq2seq/best_checkpoint.pt" "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/best_checkpoint.pt"
```

### Download Specific Checkpoint

```powershell
modal volume get chuni-checkpoints "seq2seq/model_01000.pt" "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/model_01000.pt"
```

### Download Metrics

```powershell
modal volume get chuni-checkpoints "seq2seq/metrics.csv" "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/metrics.csv"
```

---

## Listing Volume Contents

```powershell
modal volume ls chuni-checkpoints
modal volume ls chuni-checkpoints seq2seq/
```

---

## Complete Workflow Example

### First-Time Setup

```powershell
# 1. Create volume
modal volume create chuni-checkpoints

# 2. Upload all required files
modal volume put chuni-checkpoints "config.json" "config.json"
modal volume put chuni-checkpoints "log/model_04999_clean.pt" "model_04999_clean.pt"
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs/best_checkpoint.pt" "finetune_runs/best_checkpoint.pt"
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/train.jsonl" "seq2seq/train.jsonl"
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/val.jsonl" "seq2seq/val.jsonl"

# 3. Run training
modal run EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_modal.py
```

### Resuming Training

```powershell
# 1. Update config.json locally with resume path
# 2. Upload updated config
modal volume put chuni-checkpoints "config.json" "config.json"

# 3. Upload checkpoint to resume from (if not already on Modal)
modal volume put chuni-checkpoints "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/model_00250.pt" "seq2seq/model_00250.pt"

# 4. Run training
modal run EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_modal.py
```

### After Training

```powershell
# Download best model
modal volume get chuni-checkpoints "seq2seq/best_checkpoint.pt" "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/best_checkpoint.pt"

# Download metrics for analysis
modal volume get chuni-checkpoints "seq2seq/metrics.csv" "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/metrics.csv"
```

---

## Troubleshooting

### "Volume not found"
```powershell
modal volume create chuni-checkpoints
```

### "File not found on volume"
```powershell
modal volume ls chuni-checkpoints  # Check what exists
```

### Training crashes / OOM
- Reduce `batch_size` in config.json
- Modal A10G has 24GB VRAM, should handle batch_size=8

### Wrong checkpoint loaded
- Check `seq2seq_resume_path` in config.json
- Ensure `seq2seq_force_vn_init` is `false` when resuming

---

## Cost Estimation

| GPU | Cost/Hour | Speed |
|-----|-----------|-------|
| A10G | ~$1.10 | ~1-2 sec/step |
| A100 | ~$3.00 | ~0.5-1 sec/step |

**Example:** 10,000 steps at 1.5 sec/step = ~4 hours = ~$4-5

---

## Quick Reference

| Action | Command |
|--------|---------|
| Upload file | `modal volume put chuni-checkpoints "local/path" "remote/path"` |
| Download file | `modal volume get chuni-checkpoints "remote/path" "local/path"` |
| List files | `modal volume ls chuni-checkpoints` |
| Run training | `modal run EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_modal.py` |
