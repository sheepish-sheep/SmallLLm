# How Text Generation Works in the VN Pipeline

This document explains the complete text generation system, from model architecture to inference.

## Overview

The VN Pipeline uses a **seq2seq (sequence-to-sequence)** model to paraphrase text. Given a formal/stiff sentence, it generates a more casual, VN-style paraphrase.

```
Input:  "I'm not entirely convinced of your claim."
Output: "I don't really believe what you're saying."
```

## 1. Model Architecture: EncoderDecoder

The model is defined in `EncoderAndMoreInput/encoder_decoder_backup.py`.

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        EncoderDecoder                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   ENCODER    │ ───▶ │    CROSS     │ ───▶ │   DECODER    │  │
│  │ (Bidirect.)  │      │  ATTENTION   │      │  (Causal)    │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│        ▲                                            │          │
│        │                                            ▼          │
│   Source Text                                  Output Text     │
│  "paraphrase: I'm                           "I don't really    │
│   not convinced..."                          believe..."       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Encoder (Bidirectional Transformer)
- **Purpose**: Understand the input sentence fully
- **Type**: Bidirectional (can see all tokens at once)
- **Layers**: 12 transformer blocks
- **Embeddings**: Token (wte) + Position (wpe)
- **Output**: Hidden states for each input token

```python
class Encoder(nn.Module):
    # Token embedding: vocab_size → 768 dimensions
    self.transformer.wte = nn.Embedding(50304, 768)
    # Position embedding: max 128 positions
    self.transformer.wpe = nn.Embedding(128, 768)
    # 12 transformer blocks (bidirectional attention)
    self.transformer.h = nn.ModuleList([Block(...) for _ in range(12)])
    # Final layer norm
    self.transformer.ln_f = nn.LayerNorm(768)
```

#### 2. Decoder (Causal Transformer + Cross-Attention)
- **Purpose**: Generate output tokens one at a time
- **Type**: Causal (can only see previous tokens) + cross-attention to encoder
- **Layers**: 12 transformer blocks, each with:
  - Self-attention (causal mask)
  - Cross-attention (attends to encoder output)
  - Feed-forward MLP

```python
class DecoderBlock(nn.Module):
    def forward(self, x, encoder_output):
        # 1. Self-attention (causal - can only see past)
        x = x + self.attn(self.ln_1(x))

        # 2. Cross-attention (looks at encoder output)
        x = x + self.cross_attn(self.ln_3(x), encoder_output)

        # 3. Feed-forward
        x = x + self.mlp(self.ln_2(x))
        return x
```

#### 3. Cross-Attention Mechanism
The key innovation - lets the decoder "look at" the encoder output:

```python
class CrossAttention(nn.Module):
    def forward(self, x, encoder_output):
        # Query from decoder state
        q = self.c_attn_q(x)  # [batch, dec_seq, 768]

        # Key, Value from encoder output
        kv = self.c_attn_kv(encoder_output)  # [batch, enc_seq, 768*2]
        k, v = kv.split(768, dim=-1)

        # Attention: which encoder tokens are relevant?
        attn_weights = softmax(q @ k.T / sqrt(768))

        # Weighted sum of encoder values
        output = attn_weights @ v
        return self.c_proj(output)
```

### Parameter Counts

| Component | Parameters | Pretrained? |
|-----------|------------|-------------|
| Encoder (12 blocks) | ~85M | Yes (from VN-finetuned GPT) |
| Decoder self-attention | ~85M | Yes (from VN-finetuned GPT) |
| Decoder cross-attention | ~25M | No (random init) |
| Embeddings (shared) | ~38M | Yes |
| **Total** | ~276M | 75% pretrained |

## 2. Training

### Training Data Format

Training uses JSONL files with source-target pairs:

```json
{"source": "paraphrase: Our souls are intertwined.", "target": "Your heart beats with mine."}
{"source": "paraphrase: I haven't encountered this before.", "target": "This is the first I've heard of it."}
```

### Data Loading (`Seq2SeqDataLoader`)

```python
class Seq2SeqDataLoader:
    def __init__(self, pairs_path, batch_size, seq_len):
        self.enc = build_hl_encoding()  # GPT-2 tokenizer + special tokens
        self.pad_id = self.enc.eot_token  # Padding token (50256)
        self.bos_id = 50259  # Beginning of sequence

    def _load_pairs(self):
        pairs = []
        for line in jsonl_file:
            source = line["source"]
            target = line["target"]

            # Tokenize
            enc_tokens = self.enc.encode(source)
            target_tokens = self.enc.encode(target) + [self.enc.eot_token]  # Add EOS!

            pairs.append((enc_tokens, target_tokens))
        return pairs
```

### Batch Construction

For each batch, we create three tensors:

```python
# Example with one sample:
source = "paraphrase: Hello"  →  [enc_tokens]
target = "Hi there"           →  [target_tokens] + [EOS]

# Three tensors:
encoder_input = [enc_tokens]                    # What encoder sees
decoder_input = [BOS] + target_tokens[:-1]      # Teacher forcing
decoder_label = target_tokens                   # What to predict (includes EOS)
```

**Teacher Forcing**: During training, we give the decoder the correct previous tokens, not its own predictions.

### Loss Function

```python
# Cross-entropy loss with label smoothing
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # Model predictions
    labels.view(-1),              # True tokens
    ignore_index=pad_id,          # Don't penalize padding
    label_smoothing=0.1,          # Reduce overconfidence
)
```

### Training Loop

```python
for step in range(max_steps):
    # Get batch
    enc_in, dec_in, dec_out = dataloader.next_batch()

    # Forward pass
    logits, _ = model(enc_in, dec_in)

    # Compute loss
    loss = cross_entropy(logits, dec_out, ignore_index=pad_id)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

### Weight Initialization from GPT

The model is initialized from a VN-finetuned GPT checkpoint:

```python
def init_seq2seq_from_gpt(model, gpt_state):
    """Transfer weights from GPT to EncoderDecoder."""

    # Embeddings (shared between encoder and decoder)
    model.encoder.wte.weight = gpt_state["transformer.wte.weight"]
    model.decoder.wte.weight = gpt_state["transformer.wte.weight"]
    model.lm_head.weight = gpt_state["transformer.wte.weight"]

    # Position embeddings
    model.encoder.wpe.weight = gpt_state["transformer.wpe.weight"]
    model.decoder.wpe.weight = gpt_state["transformer.wpe.weight"]

    # Transformer blocks (encoder and decoder self-attention)
    for i in range(12):
        # Copy attention weights
        model.encoder.h[i].attn = gpt_state[f"transformer.h.{i}.attn"]
        model.decoder.h[i].attn = gpt_state[f"transformer.h.{i}.attn"]

        # Copy MLP weights
        model.encoder.h[i].mlp = gpt_state[f"transformer.h.{i}.mlp"]
        model.decoder.h[i].mlp = gpt_state[f"transformer.h.{i}.mlp"]

    # Cross-attention layers are NOT initialized (random)
    # They learn from scratch during training
```

**Why this helps**: The encoder and decoder start with knowledge of language/vocabulary from pretraining, so they only need to learn the paraphrasing task.

## 3. Inference (Generation)

### The Generate Loop

Generation happens one token at a time in an autoregressive loop:

```python
def generate_replacement(text, model, enc, max_len, temperature, top_k, device):
    # 1. Encode the source text
    source_tokens = enc.encode(f"paraphrase: {text}")
    enc_in = torch.tensor(source_tokens).unsqueeze(0)  # [1, src_len]

    # 2. Start decoder with BOS token
    dec_tokens = [BOS_ID]  # 50259

    # 3. Generate loop
    for _ in range(max_len):
        dec_in = torch.tensor(dec_tokens).unsqueeze(0)  # [1, dec_len]

        # Forward pass
        logits, _ = model(enc_in, dec_in)

        # Get logits for next token (last position)
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        # Apply temperature
        next_logits = next_logits / temperature

        # Top-k sampling
        if top_k > 0:
            values, indices = torch.topk(next_logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = indices.gather(-1, next_idx).item()
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        # Append to sequence
        dec_tokens.append(next_token)

        # Stop if EOS
        if next_token == EOS_ID:
            break

    # 4. Decode tokens to text
    return enc.decode(dec_tokens[1:])  # Skip BOS
```

### Sampling Parameters

| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| `temperature` | Higher = more random, Lower = more deterministic | 0.1-1.0 |
| `top_k` | Only sample from top K tokens | 30-50 |
| `max_len` | Maximum output length | 40-80 |

**Low temperature (0.1)**: Greedy, repetitive but coherent
**High temperature (0.8)**: Creative, diverse but sometimes garbled

### Visual: One Generation Step

```
Step 3: Generating 4th token
─────────────────────────────

Decoder input so far: [BOS, "I", "don't", "really"]
                       ↓
                    ┌──────────────────────┐
Encoder output ───▶ │     DECODER          │
                    │  (cross-attention    │
                    │   looks at encoder)  │
                    └──────────────────────┘
                               ↓
                    Logits: [vocab_size] probabilities
                               ↓
                    Temperature scaling (/ 0.8)
                               ↓
                    Top-k filtering (keep top 50)
                               ↓
                    Softmax → probabilities
                               ↓
                    Sample: "believe" (token 4116)
                               ↓
Decoder input now:  [BOS, "I", "don't", "really", "believe"]
```

## 4. Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. USER INPUT                                                          │
│     "Would you be so kind as to explain what occurred?"                 │
│                          ↓                                              │
│  2. SEQ2SEQ REWRITER (seq2seq_infer.py)                                │
│     - Add prefix: "paraphrase: Would you be so kind..."                │
│     - Tokenize with GPT-2 tokenizer                                    │
│     - Run through EncoderDecoder model                                 │
│     - Autoregressive generation with temperature/top_k                 │
│     - Decode tokens back to text                                       │
│                          ↓                                              │
│  3. REWRITTEN TEXT                                                      │
│     "Could you tell me what happened?"                                  │
│                          ↓                                              │
│  4. VOICE SYNTHESIS (voice_stub.py)                                    │
│     - Load VoiceCloner from Basic AI Voice Clone                       │
│     - Generate speech with Chatterbox TTS model                        │
│     - Save WAV file                                                    │
│                          ↓                                              │
│  5. OUTPUT                                                              │
│     Audio file: out/audio/voice_20260112_223000.wav                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5. Key Code Locations

| What | File | Function/Class |
|------|------|----------------|
| Model architecture | `encoder_decoder_backup.py` | `EncoderDecoder` |
| Training loop | `training/seq2seq_train_stub.py` | `train_seq2seq_loop()` |
| Data loading | `training/seq2seq_train_stub.py` | `Seq2SeqDataLoader` |
| Weight init from GPT | `training/seq2seq_train_stub.py` | `init_seq2seq_from_gpt()` |
| Inference/generation | `inference/seq2seq_infer.py` | `generate_replacement()` |
| High-level rewriter | `inference/seq2seq_infer.py` | `Seq2SeqRewriter` |
| Full pipeline | `inference/rewrite_and_speak.py` | `rewrite_and_speak()` |

## 6. Common Issues and Solutions

### Issue: Model produces repetitive output
**Cause**: Temperature too low, or model hasn't learned EOS
**Fix**: Increase temperature (0.5-0.8), ensure training data has EOS tokens

### Issue: Model produces garbage/random tokens
**Cause**: Temperature too high, or model undertrained
**Fix**: Decrease temperature, train longer, or use lower top_k

### Issue: Output doesn't stop / gets cut off
**Cause**: Model hasn't learned to emit EOS token
**Fix**: Ensure training targets include EOS token (fixed in latest code)

### Issue: Poor quality paraphrases
**Cause**: Insufficient training data or wrong source domain
**Fix**: Generate more training pairs with `generate_paraphrases.py`

## 7. Tokenization Details

The tokenizer extends GPT-2 with special tokens:

```python
# Standard GPT-2
vocab_size = 50257  # 50,000 BPE + 256 bytes + 1 EOT

# Extended for VN Pipeline
50256 = <|endoftext|>  # EOT/PAD token
50257 = <hl>           # Highlight start (unused in current training)
50258 = </hl>          # Highlight end (unused in current training)
50259 = BOS            # Beginning of sequence for decoder

# Current vocab_size = 50304 (padded for efficiency)
```

## 8. Example: Tracing Through Generation

Input: `"I haven't encountered this information before."`

```
1. Add prefix:
   "paraphrase: I haven't encountered this information before."

2. Tokenize (enc.encode):
   [1845, 10680, 7512, 25, 314, 4398, 470, 14275, 428, 1321, 878, 13]

3. Encoder forward:
   encoder_output = encoder(tokens)  # [1, 12, 768]

4. Start decoder:
   dec_tokens = [50259]  # BOS

5. Generate loop:
   Step 1: logits → sample "This" (1212)
   Step 2: logits → sample "is" (318)
   Step 3: logits → sample "the" (262)
   Step 4: logits → sample "first" (717)
   Step 5: logits → sample "I" (314)
   Step 6: logits → sample "'ve" (1053)
   Step 7: logits → sample "heard" (2982)
   Step 8: logits → sample "of" (286)
   Step 9: logits → sample "this" (428)
   Step 10: logits → sample "." (13)
   Step 11: logits → sample EOS (50256) → STOP

6. Decode:
   [1212, 318, 262, 717, 314, 1053, 2982, 286, 428, 13]
   → "This is the first I've heard of this."
```

## Summary

The VN Pipeline text generation works by:
1. **Encoding** the input text with a bidirectional transformer
2. **Decoding** autoregressively with cross-attention to the encoder
3. **Sampling** tokens based on temperature and top_k settings
4. **Stopping** when EOS is generated or max_len is reached

The model learns paraphrasing by training on (formal, casual) sentence pairs, with most weights initialized from a pretrained VN-finetuned GPT model.
