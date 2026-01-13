# VN Pipeline: Complete Flow Guide

This document explains how all pieces of the VN Pipeline connect, from data preparation to final voice output.

## Pipeline Stages Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VN PIPELINE STAGES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING STAGES (run once to create model)                                 │
│  ═══════════════════════════════════════════                                │
│                                                                             │
│  [1] Data Prep ─────▶ [2] Finetune ─────▶ [3] Seq2Seq Train                │
│      (pairs)              (VN GPT)            (EncoderDecoder)              │
│                                                                             │
│  INFERENCE STAGES (run to generate output)                                  │
│  ═════════════════════════════════════════                                  │
│                                                                             │
│  [4] Evaluation ────▶ [5] Voice ────────▶ [6] Rewrite+Voice                │
│      (metrics)           (TTS)               (full pipeline)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Data Preparation

**Purpose**: Create training pairs for the seq2seq model

### Option A: Using Claude API (Recommended)

```bash
python -m EncoderAndMoreInput.VN_Pipeline.training.generate_paraphrases
```

**File**: `training/generate_paraphrases.py`

**Flow**:
```
VN Dialogue Text
     │
     ▼
┌─────────────────────────────────────┐
│  Claude API                         │
│  "Generate a casual paraphrase of:  │
│   'I am not entirely convinced...'" │
└─────────────────────────────────────┘
     │
     ▼
Paraphrase Pairs (train.jsonl, val.jsonl)
```

**Output Format**:
```json
{"source": "paraphrase: I am not convinced.", "target": "I don't really believe it."}
{"source": "paraphrase: What occurred here?", "target": "What happened?"}
```

### Option B: Using Highlight Spans

```bash
python -m EncoderAndMoreInput.VN_Pipeline.training.data_prep_stub
```

**File**: `training/data_prep_stub.py`

Creates copy-replace pairs by highlighting random spans in VN dialogue.

## Stage 2: Fine-tune GPT on VN Text

**Purpose**: Adapt base GPT model to VN dialogue style

```bash
python -m EncoderAndMoreInput.VN_Pipeline.training.finetune_vn_stub
```

**File**: `training/finetune_vn_stub.py`

**Flow**:
```
Base GPT Checkpoint (log/model_04999_clean.pt)
     │
     ▼
┌─────────────────────────────────────┐
│  Fine-tuning Loop                   │
│  - Load VN dialogue shards          │
│  - Train with causal LM objective   │
│  - Save best checkpoint             │
└─────────────────────────────────────┘
     │
     ▼
VN-Finetuned Checkpoint (out/finetune_runs/best_checkpoint.pt)
```

**Config Settings**:
```json
{
  "base_checkpoint_path": "log/model_04999_clean.pt",
  "finetune_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs",
  "finetune_max_steps": 6000,
  "learning_rate": 1e-05
}
```

## Stage 3: Train Seq2Seq Model

**Purpose**: Train the EncoderDecoder model for paraphrasing

### Local Training

```bash
python -m EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub
```

### Modal Cloud Training (Faster)

```bash
# Upload data first
python -m EncoderAndMoreInput.VN_Pipeline.training.upload_to_modal

# Run training on Modal A10G GPU
modal run EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_modal.py
```

**File**: `training/seq2seq_train_stub.py`, `training/seq2seq_train_modal.py`

**Flow**:
```
VN-Finetuned GPT + Training Pairs
     │
     ▼
┌─────────────────────────────────────┐
│  Seq2Seq Training                   │
│  1. Initialize EncoderDecoder from  │
│     VN-finetuned GPT weights        │
│  2. Train on paraphrase pairs       │
│  3. Save checkpoints every 500 steps│
│  4. Track best by validation loss   │
└─────────────────────────────────────┘
     │
     ▼
Trained Model (out/seq2seq/best_checkpoint.pt)
```

**Config Settings**:
```json
{
  "seq2seq_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/seq2seq",
  "seq2seq_max_steps": 20000,
  "seq2seq_learning_rate": 5e-05,
  "seq2seq_force_vn_init": true,
  "max_seq_len": 128
}
```

**Key Files Created**:
- `out/seq2seq/best_checkpoint.pt` - Best model by validation loss
- `out/seq2seq/model_final.pt` - Final checkpoint
- `out/seq2seq/metrics.csv` - Training metrics log

## Stage 4: Evaluation

**Purpose**: Measure model quality

```bash
python -m EncoderAndMoreInput.VN_Pipeline.eval.seq2seq_eval_stub
```

**File**: `eval/seq2seq_eval_stub.py`

Computes metrics like:
- Validation loss
- Length ratio (output vs input)
- Sample generations

## Stage 5: Voice Synthesis

**Purpose**: Convert text to speech using voice cloning

```bash
python -m EncoderAndMoreInput.VN_Pipeline.voice_stub
```

**File**: `voice_stub.py`

### Voice Model Options

The pipeline supports two TTS models:

| Model | File | Quality | Speed | License |
|-------|------|---------|-------|---------|
| **XTTS-v2** (Recommended) | `voice_cloner_xtts.py` | Better, more natural | ~0.5x real-time | CPML (non-commercial) |
| Chatterbox | `voice_cloner.py` | Good | ~0.3x real-time | MIT |

Set `voice_model_type` in `config.json`:
```json
{
  "voice_model_type": "xtts"  // or "chatterbox"
}
```

**Flow**:
```
Text: "Hello, how are you?"
     │
     ▼
┌─────────────────────────────────────┐
│  VoiceCloner (XTTS-v2 or Chatterbox)│
│  - Load TTS model                   │
│  - Reference voice samples (.ogg)   │
│  - Generate speech audio            │
└─────────────────────────────────────┘
     │
     ▼
Audio: out/audio/voice_20260112_223000.wav
```

**Config Settings**:
```json
{
  "voice_model_type": "xtts",
  "voice_clone_root": "Basic AI Voice Clone",
  "voice_samples_dir": "Basic AI Voice Clone/voice_samples/ma",
  "voice_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/audio",
  "voice_temperature": 1.0,
  "voice_cfg_weight": 0.25
}
```

## Stage 6: Full Pipeline (Rewrite + Voice)

**Purpose**: Paraphrase text then synthesize voice

```bash
# Interactive mode
python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak -i

# Single text
python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak "Your text here"

# Via pipeline orchestrator
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages rewrite_voice
```

**File**: `inference/rewrite_and_speak.py`

**Flow**:
```
User Input: "I'm not entirely convinced of your claim."
     │
     ▼
┌─────────────────────────────────────┐
│  Seq2SeqRewriter                    │
│  1. Load EncoderDecoder model       │
│  2. Add "paraphrase: " prefix       │
│  3. Encode with tokenizer           │
│  4. Generate autoregressively       │
│  5. Decode to text                  │
└─────────────────────────────────────┘
     │
     ▼
Rewritten: "I don't really believe what you're saying."
     │
     ▼
┌─────────────────────────────────────┐
│  VoiceCloner                        │
│  1. Load TTS model                  │
│  2. Generate speech                 │
│  3. Save WAV file                   │
└─────────────────────────────────────┘
     │
     ▼
Output: {
  "input": "I'm not entirely convinced...",
  "rewritten": "I don't really believe...",
  "audio_path": "out/audio/voice_xxx.wav"
}
```

## Running the Pipeline

### Full Training Pipeline

```bash
# From repo root
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages data_prep finetune seq2seq eval
```

### Just Inference

```bash
# Interactive mode
python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak -i

# Or with pipeline
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages rewrite_voice
```

### Individual Stages

```bash
# Data prep only
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages data_prep

# Seq2seq training only
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages seq2seq

# Voice only (uses config's pipeline_voice_texts)
python -m EncoderAndMoreInput.VN_Pipeline.pipeline_stub --stages voice
```

## Configuration Reference

All settings in `config.json`:

```json
{
  // === Data Paths ===
  "vn_cleaned_path": "training_data/vn/cleaned_binary_dialogue.txt",
  "vn_shard_dir": "training_data/vn/shards",

  // === Base Model ===
  "base_checkpoint_path": "log/model_04999_clean.pt",
  "vocab_size": 50304,
  "max_seq_len": 128,

  // === Fine-tuning ===
  "finetune_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs",
  "finetune_max_steps": 6000,

  // === Seq2Seq Training ===
  "seq2seq_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/seq2seq",
  "seq2seq_checkpoint_path": "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/best_checkpoint.pt",
  "seq2seq_max_steps": 20000,
  "seq2seq_learning_rate": 5e-05,
  "seq2seq_train_from_scratch": false,
  "seq2seq_force_vn_init": true,
  "seq2seq_bos_token_id": 50259,

  // === Seq2Seq Inference ===
  "seq2seq_source_prefix": "paraphrase: ",
  "seq2seq_temperature": 0.8,
  "seq2seq_top_k": 50,
  "seq2seq_max_gen_len": 80,
  "seq2seq_len_ratio": 1.1,
  "seq2seq_min_gen_len": 8,

  // === Voice ===
  "voice_model_type": "xtts",  // "xtts" (recommended) or "chatterbox"
  "voice_clone_root": "Basic AI Voice Clone",
  "voice_samples_dir": "Basic AI Voice Clone/voice_samples/ma",
  "voice_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/audio",
  "voice_temperature": 1.0,
  "voice_cfg_weight": 0.25,

  // === Pipeline ===
  "pipeline_voice_texts": []  // Texts to process with rewrite_voice stage
}
```

## Data Flow Diagram

```
                    TRAINING DATA FLOW
                    ══════════════════

VN Dialogue Text ─────────┬─────────────────────────────┐
                          │                             │
                          ▼                             ▼
              ┌───────────────────┐         ┌───────────────────┐
              │  generate_        │         │  finetune_vn_     │
              │  paraphrases.py   │         │  stub.py          │
              │  (Claude API)     │         │  (Causal LM)      │
              └─────────┬─────────┘         └─────────┬─────────┘
                        │                             │
                        ▼                             ▼
              ┌───────────────────┐         ┌───────────────────┐
              │  train.jsonl      │         │  VN-Finetuned     │
              │  val.jsonl        │         │  GPT Checkpoint   │
              └─────────┬─────────┘         └─────────┬─────────┘
                        │                             │
                        └──────────┬──────────────────┘
                                   │
                                   ▼
                        ┌───────────────────┐
                        │  seq2seq_train_   │
                        │  stub.py          │
                        │  (EncoderDecoder) │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │  best_checkpoint  │
                        │  .pt              │
                        └───────────────────┘


                    INFERENCE DATA FLOW
                    ═══════════════════

User Input Text ──────────────────────────────────────────┐
       │                                                  │
       ▼                                                  │
┌─────────────────┐                                       │
│ Seq2SeqRewriter │◄── best_checkpoint.pt                │
│ (seq2seq_infer) │                                      │
└────────┬────────┘                                       │
         │                                                │
         ▼                                                │
Paraphrased Text ─────────────────────────────────────────┤
         │                                                │
         ▼                                                ▼
┌─────────────────┐                             ┌─────────────────┐
│   VoiceCloner   │◄── voice_samples/           │  rewrite_and_   │
│  (voice_stub)   │                             │  speak.py       │
└────────┬────────┘                             └────────┬────────┘
         │                                               │
         ▼                                               ▼
    Audio WAV File                              { input, rewritten,
                                                  audio_path }
```

## Troubleshooting

### Training Issues

| Problem | Solution |
|---------|----------|
| High initial loss (~9) | Model not initialized from pretrained. Check `seq2seq_force_vn_init: true` |
| Train/val gap too large | Overfitting. Use more data or add regularization |
| Out of memory | Reduce batch_size or use Modal cloud training |

### Inference Issues

| Problem | Solution |
|---------|----------|
| Repetitive output | Increase temperature (0.5-0.8) |
| Garbage output | Decrease temperature, use lower top_k |
| Output doesn't stop | Model hasn't learned EOS. Retrain with EOS in targets |
| Voice synthesis fails | Check `voice_samples_dir` exists with .ogg files |

### Voice Model Issues

| Problem | Solution |
|---------|----------|
| XTTS license prompt | Set `COQUI_TOS_AGREED=1` environment variable |
| XTTS PyTorch 2.6 error | voice_cloner_xtts.py patches torch.load automatically |
| Voice sounds robotic | Use XTTS (`voice_model_type: "xtts"`), increase temperature |
| XTTS slow on CPU | XTTS requires GPU for reasonable speed |
| Switch voice models | Change `voice_model_type` in config.json |

### Common Commands

```bash
# Check training progress
python -m EncoderAndMoreInput.VN_Pipeline.inference.analyze_training

# Debug weight loading
python -m EncoderAndMoreInput.VN_Pipeline.inference.debug_weights

# Test model generation
python -m EncoderAndMoreInput.VN_Pipeline.inference.test_seq2seq

# Monitor Modal training
modal volume ls chuni-checkpoints seq2seq/
```
