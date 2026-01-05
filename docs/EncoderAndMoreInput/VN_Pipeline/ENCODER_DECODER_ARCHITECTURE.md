# Encoder-Decoder Architecture

This document explains the encoder-decoder model used for the seq2seq highlight replacement task.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENCODER-DECODER MODEL                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: "The <hl>quick</hl> brown fox"                                     │
│                    │                                                        │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │                    ENCODER                          │                   │
│   │  ┌───────────────────────────────────────────────┐  │                   │
│   │  │         Bidirectional Self-Attention          │  │                   │
│   │  │   (Each token sees ALL other tokens)          │  │                   │
│   │  │                                               │  │                   │
│   │  │   [The] ←→ [<hl>] ←→ [quick] ←→ [</hl>] ←→ [brown] ←→ [fox]        │
│   │  │     ↑↓       ↑↓        ↑↓         ↑↓          ↑↓        ↑↓          │
│   │  │   Full context available at every position    │  │                   │
│   │  └───────────────────────────────────────────────┘  │                   │
│   └──────────────────────────┬──────────────────────────┘                   │
│                              │                                              │
│                    Encoder Output (hidden states)                           │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │                    DECODER                          │                   │
│   │  ┌───────────────────────────────────────────────┐  │                   │
│   │  │         Causal Self-Attention                 │  │                   │
│   │  │   (Each token sees only PAST tokens)          │  │                   │
│   │  │                                               │  │                   │
│   │  │   [The] → [fast] → [brown] → [fox]            │  │                   │
│   │  │     ↓       ↓         ↓        ↓              │  │                   │
│   │  └───────────────────────────────────────────────┘  │                   │
│   │                         │                           │                   │
│   │                         ▼                           │                   │
│   │  ┌───────────────────────────────────────────────┐  │                   │
│   │  │            Cross-Attention                    │  │                   │
│   │  │   (Decoder queries → Encoder keys/values)     │  │                   │
│   │  │                                               │  │                   │
│   │  │   Decoder: "What was in the <hl> tags?"       │  │                   │
│   │  │   Encoder: "quick"                            │  │                   │
│   │  │   Decoder: "I'll generate something similar"  │  │                   │
│   │  └───────────────────────────────────────────────┘  │                   │
│   └──────────────────────────┬──────────────────────────┘                   │
│                              │                                              │
│                              ▼                                              │
│   OUTPUT: "The fast brown fox"                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Layer-by-Layer Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETAILED DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOURCE TEXT: "The <hl>quick</hl> brown fox"                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────┐                                                │
│  │   TOKENIZATION          │                                                │
│  │   tiktoken + special    │                                                │
│  └───────────┬─────────────┘                                                │
│              │                                                              │
│              ▼                                                              │
│  Token IDs: [464, 50257, 4996, 50258, 7586, 21831]                          │
│              The  <hl>  quick  </hl>  brown   fox                           │
│              │                                                              │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                        ENCODER                                  │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │                                                                 │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  Token Embedding (wte)  +  Position Embedding (wpe)     │    │        │
│  │  │  [768-dim vector for each token]                        │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  │                          ▼                                      │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  Encoder Block 0                                        │    │        │
│  │  │  ├── LayerNorm                                          │    │        │
│  │  │  ├── Bidirectional Self-Attention (is_causal=False)     │    │        │
│  │  │  ├── Residual Connection                                │    │        │
│  │  │  ├── LayerNorm                                          │    │        │
│  │  │  ├── MLP (feed-forward)                                 │    │        │
│  │  │  └── Residual Connection                                │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  │                          ▼                                      │        │
│  │                    ... (12 blocks total) ...                    │        │
│  │                          │                                      │        │
│  │                          ▼                                      │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  Final LayerNorm (ln_f)                                 │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  └──────────────────────────┼──────────────────────────────────────┘        │
│                             │                                               │
│                 ENCODER OUTPUT: [6 x 768] tensor                            │
│                 (one 768-dim vector per input token)                        │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                        DECODER                                  │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │                                                                 │        │
│  │  TARGET INPUT: [BOS, The, fast, brown]  (teacher forcing)       │        │
│  │                                                                 │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  Token Embedding (wte)  +  Position Embedding (wpe)     │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  │                          ▼                                      │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  Decoder Block 0                                        │    │        │
│  │  │  ├── LayerNorm                                          │    │        │
│  │  │  ├── Causal Self-Attention (is_causal=True)             │    │        │
│  │  │  ├── Residual Connection                                │    │        │
│  │  │  ├── LayerNorm                                          │    │        │
│  │  │  ├── CROSS-ATTENTION ◄──────── Encoder Output           │    │        │
│  │  │  │   └── Query from decoder, Key/Value from encoder     │    │        │
│  │  │  ├── Residual Connection                                │    │        │
│  │  │  ├── LayerNorm                                          │    │        │
│  │  │  ├── MLP (feed-forward)                                 │    │        │
│  │  │  └── Residual Connection                                │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  │                    ... (12 blocks total) ...                    │        │
│  │                          │                                      │        │
│  │                          ▼                                      │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  Final LayerNorm (ln_f)                                 │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  │                          ▼                                      │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │  LM Head (project to vocab size)                        │    │        │
│  │  │  [768] → [50304] logits per position                    │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                          │                                      │        │
│  └──────────────────────────┼──────────────────────────────────────┘        │
│                             │                                               │
│                             ▼                                               │
│              LOGITS: [4 x 50304] (predict next token at each position)      │
│                             │                                               │
│                             ▼                                               │
│              Cross-Entropy Loss with target: [The, fast, brown, fox]        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cross-Attention Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CROSS-ATTENTION MECHANISM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  The decoder asks: "What should I output next?"                             │
│  Cross-attention answers: "Look at the encoder's understanding of input"   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                                                                 │        │
│  │   DECODER (generating "fast")        ENCODER (saw "<hl>quick</hl>")     │
│  │            │                                    │               │        │
│  │            ▼                                    ▼               │        │
│  │   ┌─────────────────┐               ┌─────────────────┐         │        │
│  │   │  Query Matrix   │               │  Key Matrix     │         │        │
│  │   │  W_q * decoder  │               │  W_k * encoder  │         │        │
│  │   └────────┬────────┘               └────────┬────────┘         │        │
│  │            │                                 │                  │        │
│  │            └────────────┬───────────────────┘                   │        │
│  │                         │                                       │        │
│  │                         ▼                                       │        │
│  │              ┌─────────────────────┐                            │        │
│  │              │  Attention Scores   │                            │        │
│  │              │  Q @ K^T / sqrt(d)  │                            │        │
│  │              │                     │                            │        │
│  │              │  "How relevant is   │                            │        │
│  │              │   each encoder      │                            │        │
│  │              │   position to the   │                            │        │
│  │              │   current decoder   │                            │        │
│  │              │   position?"        │                            │        │
│  │              └──────────┬──────────┘                            │        │
│  │                         │                                       │        │
│  │                         ▼                                       │        │
│  │              ┌─────────────────────┐                            │        │
│  │              │      Softmax        │                            │        │
│  │              └──────────┬──────────┘                            │        │
│  │                         │                                       │        │
│  │                         ▼                                       │        │
│  │   Attention weights:  [0.1, 0.3, 0.4, 0.1, 0.05, 0.05]          │        │
│  │                        The  <hl> quick </hl> brown  fox         │        │
│  │                              ↑     ↑                            │        │
│  │                         Highest attention on highlighted word!  │        │
│  │                                                                 │        │
│  │                         │                                       │        │
│  │                         ▼                                       │        │
│  │              ┌─────────────────────┐                            │        │
│  │              │  Value Matrix       │                            │        │
│  │              │  W_v * encoder      │                            │        │
│  │              └──────────┬──────────┘                            │        │
│  │                         │                                       │        │
│  │                         ▼                                       │        │
│  │              ┌─────────────────────┐                            │        │
│  │              │  Weighted Sum       │                            │        │
│  │              │  attn @ V           │                            │        │
│  │              └──────────┬──────────┘                            │        │
│  │                         │                                       │        │
│  │                         ▼                                       │        │
│  │              Output: Information from encoder, focused on       │        │
│  │                      the highlighted region                     │        │
│  │                                                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                     DATA PREPARATION                             │      │
│   ├──────────────────────────────────────────────────────────────────┤      │
│   │                                                                  │      │
│   │   Original Text: "The door creaked open."                        │      │
│   │                            │                                     │      │
│   │                            ▼                                     │      │
│   │   data_prep_stub.py adds highlight tags                          │      │
│   │                            │                                     │      │
│   │                            ▼                                     │      │
│   │   Source: "SOURCE: The door <hl>creaked</hl> open. => TARGET: "  │      │
│   │   Target: "The door creaked open."                               │      │
│   │                            │                                     │      │
│   │                            ▼                                     │      │
│   │   Saved to: train.jsonl, val.jsonl                               │      │
│   │                                                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                     TRAINING STEP                                │      │
│   ├──────────────────────────────────────────────────────────────────┤      │
│   │                                                                  │      │
│   │   for step in range(max_steps):                                  │      │
│   │       │                                                          │      │
│   │       ├── 1. Load batch from dataloader                          │      │
│   │       │      enc_in, dec_in, dec_out = dataloader.next_batch()   │      │
│   │       │                                                          │      │
│   │       ├── 2. Forward pass                                        │      │
│   │       │      logits, _ = model(enc_in, dec_in)                   │      │
│   │       │                                                          │      │
│   │       ├── 3. Compute loss                                        │      │
│   │       │      loss = cross_entropy(logits, dec_out)               │      │
│   │       │                                                          │      │
│   │       ├── 4. Backward pass                                       │      │
│   │       │      loss.backward()                                     │      │
│   │       │                                                          │      │
│   │       ├── 5. Update weights                                      │      │
│   │       │      optimizer.step()                                    │      │
│   │       │                                                          │      │
│   │       ├── 6. Validation (every N steps)                          │      │
│   │       │      if step % val_interval == 0:                        │      │
│   │       │          compute val_loss                                │      │
│   │       │          save to metrics.csv                             │      │
│   │       │                                                          │      │
│   │       └── 7. Save checkpoint (every M steps)                     │      │
│   │              save model state                                    │      │
│   │                                                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE (GENERATION)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: "The <hl>quick</hl> brown fox"                                     │
│                                                                             │
│   Step 1: Encode source (done once)                                         │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  enc_output = encoder(source_tokens)                            │       │
│   │  # Shape: [1, 6, 768]                                           │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   Step 2: Auto-regressive generation                                        │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   dec_tokens = [BOS]                                            │       │
│   │                                                                 │       │
│   │   Iteration 1:                                                  │       │
│   │   ┌─────────────────────────────────────────────────────────┐   │       │
│   │   │  logits = decoder([BOS], enc_output)                    │   │       │
│   │   │  next_token = sample(logits[-1])  →  "The"              │   │       │
│   │   │  dec_tokens = [BOS, "The"]                              │   │       │
│   │   └─────────────────────────────────────────────────────────┘   │       │
│   │                                                                 │       │
│   │   Iteration 2:                                                  │       │
│   │   ┌─────────────────────────────────────────────────────────┐   │       │
│   │   │  logits = decoder([BOS, "The"], enc_output)             │   │       │
│   │   │  next_token = sample(logits[-1])  →  "fast"             │   │       │
│   │   │  dec_tokens = [BOS, "The", "fast"]                      │   │       │
│   │   └─────────────────────────────────────────────────────────┘   │       │
│   │                                                                 │       │
│   │   Iteration 3:                                                  │       │
│   │   ┌─────────────────────────────────────────────────────────┐   │       │
│   │   │  logits = decoder([BOS, "The", "fast"], enc_output)     │   │       │
│   │   │  next_token = sample(logits[-1])  →  "brown"            │   │       │
│   │   │  dec_tokens = [BOS, "The", "fast", "brown"]             │   │       │
│   │   └─────────────────────────────────────────────────────────┘   │       │
│   │                                                                 │       │
│   │   ... continues until EOS or max_len ...                        │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   OUTPUT: "The fast brown fox"                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why Training From Scratch Matters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION COMPARISON                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ❌ WRONG: Initialize from pre-trained LM                                  │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   Self-Attention: ████████████ (strong, from LM)               │       │
│   │   Cross-Attention: ░░░░░░░░░░░ (random noise)                  │       │
│   │                                                                 │       │
│   │   Result: Model ignores cross-attention, just generates like LM│       │
│   │           Output has NO connection to input!                   │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   ✅ CORRECT: Train from scratch                                            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   Self-Attention: ░░░░░░░░░░░ (learning)                       │       │
│   │   Cross-Attention: ░░░░░░░░░░░ (learning together)             │       │
│   │                                                                 │       │
│   │   Result: All layers learn together                            │       │
│   │           Cross-attention learns to read encoder output        │       │
│   │           Output is conditioned on input!                      │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Model Dimensions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 50304 | Vocabulary size (GPT-2 + special tokens) |
| `block_size` | 512 | Maximum sequence length |
| `n_embd` | 768 | Embedding dimension |
| `n_head` | 12 | Number of attention heads |
| `n_layer` | 12 | Number of transformer blocks |

## File Locations

| File | Purpose |
|------|---------|
| `EncoderAndMoreInput/encoder_decoder_backup.py` | Model architecture |
| `EncoderAndMoreInput/VN_Pipeline/training/seq2seq_train_stub.py` | Training loop |
| `EncoderAndMoreInput/VN_Pipeline/inference/seq2seq_infer.py` | Inference/generation |
| `EncoderAndMoreInput/VN_Pipeline/training/data_prep_stub.py` | Data preparation |
