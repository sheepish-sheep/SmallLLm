modal volume ls chuni-checkpoints
  modal volume ls chuni-checkpoints seq2seq/# GPT Architecture Components Explained

This document explains the MLP, Block, GPTConfig, and GPT classes that form a complete GPT (Generative Pre-trained Transformer) model.

---

## 1. MLP Class (Multi-Layer Perceptron)

### Purpose
The MLP is a feedforward neural network that processes the output from the attention mechanism. It's a standard component in Transformer blocks that adds non-linearity and allows the model to learn complex transformations.

### Code Breakdown

#### `__init__` Method (Lines 44-49)

```python
def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate="tanh")
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
```

**Explanation:**
- **Line 46**: `c_fc` (fully connected) - Expands embedding dimension from `n_embd` to `4 * n_embd`
  - Example: 768 → 3072 (4× expansion)
  - This expansion allows the model to learn richer representations
  
- **Line 47**: `GELU` activation function - Gaussian Error Linear Unit
  - Non-linear activation that smooths the ReLU function
  - `approximate="tanh"` uses a faster tanh-based approximation
  - Formula: `GELU(x) = x * Φ(x)` where Φ is the CDF of standard normal
  
- **Line 48**: `c_proj` (projection) - Projects back from `4 * n_embd` to `n_embd`
  - Example: 3072 → 768 (back to original size)
  - This compression forces the model to learn efficient representations
  
- **Line 49**: Special initialization flag for NanoGPT-style weight scaling

#### `forward` Method (Lines 51-55)

```python
def forward(self, x):
    x = self.c_fc(x)      # Expand: (B, T, C) → (B, T, 4*C)
    x = self.gelu(x)      # Apply non-linearity
    x = self.c_proj(x)    # Project back: (B, T, 4*C) → (B, T, C)
    return x
```

**Matrix Operations Flow:**

```
Input x: (B, T, C)
    │
    ├─→ Linear(c_fc: C → 4C)
    │         │
    │         ↓
    │   Intermediate: (B, T, 4*C)
    │         │
    │         ├─→ GELU Activation
    │         │   (element-wise non-linearity)
    │         │
    │         ↓
    │   After GELU: (B, T, 4*C)
    │         │
    │         ├─→ Linear(c_proj: 4C → C)
    │         │
    │         ↓
    Output: (B, T, C)
```

**Visual Diagram:**

```
Input: (B=2, T=10, C=768)
┌────────────────────────────────────────┐
│ Batch 1: 10 tokens, 768 dims each      │
│ ┌──────┐ ┌──────┐ ... ┌──────┐        │
│ │ 768  │ │ 768  │     │ 768  │        │
│ └──────┘ └──────┘     └──────┘        │
└────────────────────────────────────────┘
         ↓ Linear(c_fc)
Expanded: (B=2, T=10, 4*C=3072)
┌────────────────────────────────────────┐
│ Batch 1: 10 tokens, 3072 dims each     │
│ ┌────────┐ ┌────────┐ ... ┌────────┐ │
│ │ 3072   │ │ 3072   │     │ 3072   │ │
│ └────────┘ └────────┘     └────────┘ │
└────────────────────────────────────────┘
         ↓ GELU (element-wise)
After GELU: (B=2, T=10, 3072)
┌────────────────────────────────────────┐
│ Same shape, but values transformed     │
│ by GELU activation function            │
└────────────────────────────────────────┘
         ↓ Linear(c_proj)
Output: (B=2, T=10, C=768)
┌────────────────────────────────────────┐
│ Batch 1: 10 tokens, 768 dims each      │
│ ┌──────┐ ┌──────┐ ... ┌──────┐        │
│ │ 768  │ │ 768  │     │ 768  │        │
│ └──────┘ └──────┘     └──────┘        │
└────────────────────────────────────────┘
```

**Why 4× Expansion?**
- Provides more capacity for learning complex transformations
- Standard in Transformer architectures (GPT, BERT, etc.)
- The bottleneck (back to original size) forces efficient learning

---

## 2. Block Class (Transformer Block)

### Purpose
A complete Transformer block that combines self-attention and MLP with residual connections and layer normalization. This is the fundamental building block of the GPT model.

### Code Breakdown

#### `__init__` Method (Lines 59-64)

```python
def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
```

**Explanation:**
- **Line 61**: `ln_1` - First layer normalization (before attention)
  - Normalizes inputs to attention for training stability
  
- **Line 62**: `attn` - Causal self-attention module
  - Allows tokens to attend to previous tokens
  
- **Line 63**: `ln_2` - Second layer normalization (before MLP)
  - Normalizes inputs to MLP
  
- **Line 64**: `mlp` - Multi-layer perceptron
  - Processes attention output

#### `forward` Method (Lines 66-69)

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))  # Residual connection #1
    x = x + self.mlp(self.ln_2(x))   # Residual connection #2
    return x
```

**This implements Pre-LayerNorm architecture with residual connections!**

**Flow Diagram:**

```
Input x: (B, T, C)
    │
    ├─→ LayerNorm(ln_1)
    │         │
    │         ↓
    │   Normalized: (B, T, C)
    │         │
    │         ├─→ CausalSelfAttention
    │         │         │
    │         │         ↓
    │         │   Attention Output: (B, T, C)
    │         │         │
    │         └─────────┘
    │         │
    │         ↓
    │   Residual Add: x + attention_output
    │         │
    │         ↓
    │   After Attention: (B, T, C)
    │         │
    │         ├─→ LayerNorm(ln_2)
    │         │         │
    │         │         ↓
    │         │   Normalized: (B, T, C)
    │         │         │
    │         │         ├─→ MLP
    │         │         │         │
    │         │         │         ↓
    │         │         │   MLP Output: (B, T, C)
    │         │         │         │
    │         │         └─────────┘
    │         │         │
    │         └─────────┘
    │         │
    │         ↓
    │   Residual Add: x + mlp_output
    │         │
    │         ↓
    Output: (B, T, C)
```

**Visual Representation:**

```
┌─────────────────────────────────────────────────────────┐
│                    Transformer Block                     │
│                                                          │
│  Input x: (B, T, C)                                     │
│      │                                                   │
│      ├─────────────────────────────────┐               │
│      │                                 │               │
│      ▼                                 │               │
│  ┌─────────────┐                       │               │
│  │ LayerNorm 1 │                       │               │
│  └──────┬──────┘                       │               │
│         │                               │               │
│         ▼                               │               │
│  ┌──────────────────┐                  │               │
│  │ Self-Attention   │                  │               │
│  └──────┬───────────┘                  │               │
│         │                               │               │
│         └───────────┬───────────────────┘               │
│                     │                                   │
│                     ▼                                   │
│              x + attention_output                       │
│                     │                                   │
│                     ▼                                   │
│              ┌─────────────┐                            │
│              │ LayerNorm 2 │                            │
│              └──────┬──────┘                            │
│                     │                                   │
│                     ▼                                   │
│              ┌──────────┐                              │
│              │   MLP    │                              │
│              └──────┬───┘                              │
│                     │                                   │
│                     └───────────┬──────────────────────┘
│                                 │                       │
│                                 ▼                       │
│                          x + mlp_output                │
│                                 │                       │
│                                 ▼                       │
│                          Output: (B, T, C)             │
└─────────────────────────────────────────────────────────┘
```

**Key Concepts:**

1. **Residual Connections** (`x + ...`)
   - Allows gradients to flow directly through the network
   - Enables training of very deep networks
   - Preserves information from previous layers

2. **Pre-LayerNorm Architecture**
   - LayerNorm is applied BEFORE the sub-layer (attention/MLP)
   - More stable training than Post-LayerNorm
   - Standard in modern Transformers

3. **Two Sub-Layers**
   - **Attention**: Captures relationships between tokens
   - **MLP**: Processes and transforms the information

---

## 3. GPTConfig Class (Configuration)

### Purpose
A dataclass that stores all hyperparameters for the GPT model. Makes it easy to configure different model sizes.

### Code Breakdown (Lines 71-77)

```python
@dataclass
class GPTConfig:
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50257     # vocabulary size
    n_layer: int = 12           # number of transformer blocks
    n_head: int = 12            # number of attention heads
    n_embd: int = 768           # embedding dimension
```

**Explanation:**
- **block_size**: Maximum sequence length the model can process
  - GPT-2 uses 1024 tokens
  - Longer sequences require more memory
  
- **vocab_size**: Number of unique tokens in the vocabulary
  - 50,257 = 50,000 BPE merges + 256 byte tokens + 1 special token
  - Each token gets an embedding vector
  
- **n_layer**: Number of Transformer blocks stacked
  - More layers = deeper model = more capacity
  - GPT-2 small uses 12 layers
  
- **n_head**: Number of parallel attention heads
  - Each head learns different patterns
  - 12 heads × 64 dims = 768 total embedding dim
  
- **n_embd**: Embedding dimension
  - Size of token representations
  - 768 is standard for GPT-2 small

**Example Configurations:**

| Model | n_layer | n_head | n_embd | Parameters |
|-------|---------|--------|--------|------------|
| GPT-2 Small | 12 | 12 | 768 | ~124M |
| GPT-2 Medium | 24 | 16 | 1024 | ~355M |
| GPT-2 Large | 36 | 20 | 1280 | ~774M |

---

## 4. GPT Class (Complete Model)

### Purpose
The complete GPT model that combines embeddings, multiple transformer blocks, and a language modeling head to generate text.

### Code Breakdown

#### `__init__` Method (Lines 81-97)

```python
def __init__(self, config):
    super().__init__()
    self.config = config
    
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    # weight sharing scheme
    self.transformer.wte.weight = self.lm_head.weight
    
    # init params
    self.apply(self._init_weights)
```

**Component Explanation:**

1. **wte (Word Token Embedding)** - Line 86
   - Maps token IDs to embedding vectors
   - Shape: `(vocab_size, n_embd)` = `(50257, 768)`
   - Each token gets a 768-dimensional vector

2. **wpe (Word Position Embedding)** - Line 87
   - Maps position indices to embedding vectors
   - Shape: `(block_size, n_embd)` = `(1024, 768)`
   - Adds positional information to tokens

3. **h (Transformer Blocks)** - Line 88
   - List of `n_layer` Transformer blocks
   - Each block processes the sequence
   - Blocks are stacked sequentially

4. **ln_f (Final LayerNorm)** - Line 89
   - Final normalization before output
   - Stabilizes the final representations

5. **lm_head (Language Modeling Head)** - Line 91
   - Projects embeddings to vocabulary logits
   - Shape: `(n_embd, vocab_size)` = `(768, 50257)`
   - Outputs probability distribution over vocabulary

6. **Weight Sharing** - Line 94
   - Shares weights between input embeddings and output head
   - Reduces parameters and improves efficiency
   - Common in language models

#### `_init_weights` Method (Lines 99-108)

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Explanation:**
- **Linear Layers**: Initialize with small random values (std=0.02)
- **Scaled Init**: Attention/MLP output projections get smaller std based on depth
  - Formula: `std = 0.02 * (2 * n_layer)^(-0.5)`
  - For 12 layers: `std = 0.02 * (24)^(-0.5) ≈ 0.004`
  - Prevents activations from growing too large in deep networks
- **Biases**: Set to zero
- **Embeddings**: Initialize with std=0.02

#### `forward` Method (Lines 110-128)

```python
def forward(self, idx, target=None):
    B, T = idx.size()
    assert T <= self.config.block_size
    
    # Embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(idx)
    x = tok_emb + pos_emb
    
    # Transformer blocks
    for block in self.transformer.h:
        x = block(x)
    
    # Final output
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    
    # Loss computation
    loss = None
    if target is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    return logits, loss
```

**Complete Forward Pass Flow:**

```
Input idx: (B, T) - token indices
    │
    ├─→ Create position indices [0, 1, 2, ..., T-1]
    │         │
    │         ├─→ Position Embedding (wpe)
    │         │         │
    │         │         ↓
    │         │   pos_emb: (T, n_embd) → (B, T, n_embd)
    │         │
    │         └─→ Token Embedding (wte)
    │                   │
    │                   ↓
    │             tok_emb: (B, T, n_embd)
    │                   │
    │                   └─→ Add together
    │                         │
    │                         ↓
    │                   x: (B, T, n_embd)
    │                         │
    │                         ├─→ Block 1
    │                         │         │
    │                         │         ↓
    │                         │   (B, T, n_embd)
    │                         │         │
    │                         ├─→ Block 2
    │                         │         │
    │                         │         ↓
    │                         │   (B, T, n_embd)
    │                         │         │
    │                         ├─→ ... (n_layer blocks)
    │                         │         │
    │                         │         ↓
    │                         │   (B, T, n_embd)
    │                         │         │
    │                         └─→ Final LayerNorm
    │                                   │
    │                                   ↓
    │                             (B, T, n_embd)
    │                                   │
    │                                   ├─→ Language Model Head
    │                                   │         │
    │                                   │         ↓
    │                                   │   logits: (B, T, vocab_size)
    │                                   │         │
    │                                   │         ├─→ If target provided:
    │                                   │         │         │
    │                                   │         │         ├─→ Reshape to (B*T, vocab_size)
    │                                   │         │         ├─→ Compute Cross-Entropy Loss
    │                                   │         │         │
    │                                   │         │         ↓
    │                                   │         │   loss: scalar
    │                                   │         │
    │                                   └─────────┘
    │                                   │
    │                                   ↓
    │                            return logits, loss
```

**Visual Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│                        GPT Model                             │
│                                                              │
│  Input: Token IDs (B, T)                                    │
│      │                                                        │
│      ├─→ Token Embedding (wte)                              │
│      │         │                                              │
│      │         ↓                                              │
│      │   Token Embeddings: (B, T, 768)                      │
│      │         │                                              │
│      ├─→ Position Embedding (wpe)                            │
│      │         │                                              │
│      │         ↓                                              │
│      │   Position Embeddings: (B, T, 768)                    │
│      │         │                                              │
│      └─→ Add Together                                         │
│              │                                                │
│              ↓                                                │
│        Combined: (B, T, 768)                                  │
│              │                                                │
│              ├─→ Block 1 (Attention + MLP)                   │
│              │         │                                      │
│              │         ↓                                      │
│              │   (B, T, 768)                                 │
│              │         │                                      │
│              ├─→ Block 2                                      │
│              │         │                                      │
│              │         ↓                                      │
│              │   (B, T, 768)                                 │
│              │         │                                      │
│              ├─→ ... (10 more blocks)                        │
│              │         │                                      │
│              │         ↓                                      │
│              │   (B, T, 768)                                 │
│              │         │                                      │
│              └─→ Final LayerNorm                              │
│                      │                                        │
│                      ↓                                        │
│                (B, T, 768)                                    │
│                      │                                        │
│                      ├─→ Language Model Head                  │
│                      │         │                              │
│                      │         ↓                              │
│                      │   Logits: (B, T, 50257)               │
│                      │         │                              │
│                      │         ├─→ Softmax (implicit)         │
│                      │         │                              │
│                      │         ↓                              │
│                      │   Probabilities over vocabulary        │
│                      │                                        │
│                      └─→ Loss (if target provided)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Step-by-Step Example:**

Let's trace through with **B=2, T=5, vocab_size=50257, n_embd=768, n_layer=12**:

1. **Input**: `idx = (2, 5)` - 2 sequences, 5 tokens each
   ```
   Sequence 1: [101, 234, 567, 890, 1234]
   Sequence 2: [456, 789, 12, 345, 678]
   ```

2. **Token Embedding**: `tok_emb = (2, 5, 768)`
   - Each token ID → 768-dim vector

3. **Position Embedding**: `pos_emb = (2, 5, 768)`
   - Position [0,1,2,3,4] → 768-dim vectors
   - Broadcasted to batch dimension

4. **Combine**: `x = tok_emb + pos_emb = (2, 5, 768)`
   - Element-wise addition

5. **Block Processing**: 
   - Block 1: `(2, 5, 768) → (2, 5, 768)`
   - Block 2: `(2, 5, 768) → (2, 5, 768)`
   - ... (12 blocks total)

6. **Final LayerNorm**: `(2, 5, 768) → (2, 5, 768)`

7. **Language Model Head**: `(2, 5, 768) → (2, 5, 50257)`
   - Each token position gets logits for all vocabulary tokens

8. **Loss Computation** (if target provided):
   - Reshape: `(2, 5, 50257) → (10, 50257)`
   - Cross-entropy: Compare predicted vs actual next tokens

---

## Key Concepts Summary

### 1. **Embeddings**
- **Token Embeddings**: Learn semantic meaning of words
- **Position Embeddings**: Encode position information
- **Combined**: `token_emb + pos_emb` gives context-aware representations

### 2. **Transformer Blocks**
- Each block refines the representations
- Attention captures relationships
- MLP processes information
- Residual connections enable deep networks

### 3. **Language Modeling**
- Predicts next token probability distribution
- Trained to maximize likelihood of training data
- Can generate text by sampling from predictions

### 4. **Weight Sharing**
- Input and output embeddings share weights
- Reduces parameters
- Improves efficiency

---

## Matrix Dimensions Summary

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| Token Embedding | (B, T) | (B, T, n_embd) | Lookup table |
| Position Embedding | (T,) | (B, T, n_embd) | Broadcasted |
| Combined Embeddings | - | (B, T, n_embd) | Element-wise add |
| Block (each) | (B, T, n_embd) | (B, T, n_embd) | 12 blocks total |
| Final LayerNorm | (B, T, n_embd) | (B, T, n_embd) | Normalization |
| Language Model Head | (B, T, n_embd) | (B, T, vocab_size) | Logits |
| Loss Computation | (B, T, vocab_size) | scalar | Cross-entropy |

---

## Training vs Inference

### Training Mode
- Input: `idx` (current tokens) and `target` (next tokens)
- Forward pass computes loss
- Backpropagation updates weights

### Inference Mode
- Input: `idx` (prompt tokens), no `target`
- Forward pass generates logits
- Sample from logits to generate next token
- Append to sequence and repeat

---

## Additional Methods

### `configure_optimizers` Method

The GPT class also includes a `configure_optimizers` method (lines 130-153) that sets up the optimizer with selective weight decay. This is a critical component for effective training.

**For detailed explanation**, see: [`OPTIMIZER_EXPLANATION.md`](OPTIMIZER_EXPLANATION.md)

**Brief Summary:**
- Separates parameters into decay and no-decay groups
- 2D parameters (weights, embeddings) get weight decay
- 1D parameters (biases, LayerNorm) get no decay
- Creates AdamW optimizer with fused option if available

---

This architecture enables GPT to understand context, learn language patterns, and generate coherent text by processing sequences through multiple transformer blocks that progressively refine token representations.

