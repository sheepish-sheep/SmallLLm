# Causal Self-Attention Explained

## Overview
This document explains the `CausalSelfAttention` class, which implements multi-head causal self-attention - the core mechanism in Transformer models like GPT.

## Code Breakdown

### `__init__` Method (Lines 14-24)

```python
def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    self.n_head = config.n_head
    self.n_embd = config.n_embd
```

**Explanation:**
- **Line 16**: Ensures `n_embd` is divisible by `n_head` so each head gets equal dimensions
- **Line 18**: Creates a linear layer that projects input to 3× the embedding size (for Q, K, V concatenated)
- **Line 20**: Output projection layer to transform attention output back to embedding size
- **Line 21**: Special initialization flag for NanoGPT-style scaling
- **Lines 23-24**: Store number of heads and embedding dimension

### `forward` Method (Lines 26-40)

The forward pass performs multi-head self-attention with these steps:

#### Step 1: Input Preparation (Line 27)
```python
B, T, C = x.size()
```
- **B**: Batch size (number of sequences processed together)
- **T**: Sequence length (number of tokens)
- **C**: Embedding dimension (`n_embd`)

**Input Shape**: `(B, T, C)`

Example: `(2, 10, 768)` = 2 sequences, 10 tokens each, 768-dimensional embeddings

---

#### Step 2: Compute Q, K, V (Lines 32-33)
```python
qkv = self.c_attn(x)
q, k, v = qkv.split(self.n_embd, dim=2)
```

**Matrix Operation:**

```
Input x: (B, T, C)
         ↓ Linear(c_attn: C → 3C)
QKV: (B, T, 3*C)
         ↓ Split into 3 chunks
Q: (B, T, C)  K: (B, T, C)  V: (B, T, C)
```

**Visual Diagram:**

```
Input Tensor x:
┌─────────────────────┐
│ (B, T, C)           │  Example: (2, 10, 768)
│                     │
│  ┌─────┐ ┌─────┐   │
│  │ t₁  │ │ t₂  │...│  Each token is C-dim vector
│  └─────┘ └─────┘   │
└─────────────────────┘
         ↓
    Linear Layer
    (Weight: C × 3C)
         ↓
QKV Tensor:
┌──────────────────────────────┐
│ (B, T, 3*C)                  │  Example: (2, 10, 2304)
│                              │
│  ┌──────────┐ ┌──────────┐  │
│  │ QKV₁     │ │ QKV₂     │  │  Each token now 3C-dim
│  └──────────┘ └──────────┘  │
└──────────────────────────────┘
         ↓
    Split into 3
         ↓
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Q:      │  │ K:      │  │ V:      │
    │(B,T,C)  │  │(B,T,C)  │  │(B,T,C)  │
    └─────────┘  └─────────┘  └─────────┘
```

**Why concatenate Q, K, V?**
- More efficient: One matrix multiplication instead of three
- Shared computation across heads
- Common optimization in modern implementations

---

#### Step 3: Reshape for Multi-Head Attention (Lines 34-36)
```python
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

**Transformation:**

```
Before: (B, T, C)
         ↓ view(B, T, n_head, head_dim)
After view: (B, T, n_head, head_dim)  where head_dim = C // n_head
         ↓ transpose(1, 2)
Final: (B, n_head, T, head_dim)
```

**Visual Diagram:**

**Example: B=2, T=10, C=768, n_head=12, head_dim=64**

```
Q (original): (B=2, T=10, C=768)
┌────────────────────────────────────────┐
│ Batch 1: 10 tokens, 768 dims each      │
│ ┌──────┐ ┌──────┐ ... ┌──────┐        │
│ │ 768  │ │ 768  │     │ 768  │        │
│ └──────┘ └──────┘     └──────┘        │
│                                        │
│ Batch 2: 10 tokens, 768 dims each      │
│ ┌──────┐ ┌──────┐ ... ┌──────┐        │
│ │ 768  │ │ 768  │     │ 768  │        │
│ └──────┘ └──────┘     └──────┘        │
└────────────────────────────────────────┘
         ↓ view(2, 10, 12, 64)
Q after view: (B=2, T=10, n_head=12, head_dim=64)
┌────────────────────────────────────────┐
│ Batch 1:                               │
│ Token 1: ┌───┐┌───┐...┌───┐           │
│         │64 ││64 │   │64 │ (12 heads)│
│         └───┘└───┘   └───┘           │
│ Token 2: ┌───┐┌───┐...┌───┐           │
│         │64 ││64 │   │64 │           │
│         └───┘└───┘   └───┘           │
│ ... (10 tokens total)                 │
└────────────────────────────────────────┘
         ↓ transpose(1, 2)
Q final: (B=2, n_head=12, T=10, head_dim=64)
┌────────────────────────────────────────┐
│ Batch 1:                               │
│ Head 1:  ┌───┐┌───┐...┌───┐           │
│         │64 ││64 │   │64 │ (10 tokens)│
│         └───┘└───┘   └───┘           │
│ Head 2:  ┌───┐┌───┐...┌───┐           │
│         │64 ││64 │   │64 │           │
│         └───┘└───┘   └───┘           │
│ ... (12 heads total)                  │
└────────────────────────────────────────┘
```

**Why reshape this way?**
- Separates each head's computation
- Allows parallel processing of all heads
- Standard format for multi-head attention

---

#### Step 4: Scaled Dot-Product Attention (Line 37)
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**This is the core self-attention mechanism!**

**What happens inside:**

1. **Compute Attention Scores**: Q × K^T
2. **Scale**: Divide by √(head_dim)
3. **Apply Causal Mask**: Prevent attending to future tokens
4. **Softmax**: Convert scores to probabilities
5. **Weighted Sum**: Multiply attention weights × V

**Mathematical Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- `d_k` = head_dim (dimension of key vectors)
- `is_causal=True` means we mask future positions

**Detailed Matrix Operations:**

**Step 4.1: Compute Attention Scores (QK^T)**

```
Q: (B, n_head, T, head_dim)
K: (B, n_head, T, head_dim)
         ↓
    Q @ K^T
    (transpose last 2 dims of K)
         ↓
Scores: (B, n_head, T, T)
```

**Visual Diagram:**

```
For a single head (B=1, head=1):

Q: (T=10, head_dim=64)              K: (T=10, head_dim=64)
┌──────────────┐                    ┌──────────────┐
│ t₁: [64 dim] │                    │ t₁: [64 dim] │
│ t₂: [64 dim] │                    │ t₂: [64 dim] │
│ t₃: [64 dim] │                    │ t₃: [64 dim] │
│ ...          │                    │ ...          │
│ t₁₀: [64 dim]│                    │ t₁₀: [64 dim]│
└──────────────┘                    └──────────────┘
         ↓                                   ↓
    Q @ K^T                         (transpose K)
         ↓
Scores: (T=10, T=10)
┌────────────────────────────────────────┐
│        t₁  t₂  t₃  ...  t₁₀           │
│   t₁ [s₁₁ s₁₂ s₁₃  ...  s₁₁₀]         │
│   t₂ [s₂₁ s₂₂ s₂₃  ...  s₂₁₀]         │
│   t₃ [s₃₁ s₃₂ s₃₃  ...  s₃₁₀]         │
│   ...                                  │
│   t₁₀[s₁₀₁ s₁₀₂ ...  ...  s₁₀₁₀]      │
└────────────────────────────────────────┘
```

**What each score means:**
- `s₁₂` = how much token 1 should attend to token 2
- Higher score = more attention
- Scores are similarity measures between query and key vectors

**Step 4.2: Scaling**

```
Scores = Scores / √(head_dim)
```

Dividing by √64 = 8 makes the gradients more stable during training.

**Step 4.3: Causal Masking (is_causal=True)**

```
Before masking:                    After causal mask:
┌────────────────────────┐         ┌────────────────────────┐
│ t₁  t₂  t₃  t₄  t₅     │         │ t₁  t₂  t₃  t₄  t₅     │
│ t₁ [●  ●  ●  ●  ● ]    │         │ t₁ [●  ⊗  ⊗  ⊗  ⊗ ]    │
│ t₂ [●  ●  ●  ●  ● ]    │   →     │ t₂ [●  ●  ⊗  ⊗  ⊗ ]    │
│ t₃ [●  ●  ●  ●  ● ]    │         │ t₃ [●  ●  ●  ⊗  ⊗ ]    │
│ t₄ [●  ●  ●  ●  ● ]    │         │ t₄ [●  ●  ●  ●  ⊗ ]    │
│ t₅ [●  ●  ●  ●  ● ]    │         │ t₅ [●  ●  ●  ●  ● ]    │
└────────────────────────┘         └────────────────────────┘
                                   ● = allowed
                                   ⊗ = masked (-inf)

Rule: Token at position i can only attend to positions ≤ i
```

This prevents "looking into the future" - crucial for autoregressive models.

**Step 4.4: Softmax**

```
Masked Scores → Softmax → Attention Weights (probabilities)
```

Each row sums to 1.0 (probability distribution).

```
Attention Weights (after softmax):
┌────────────────────────────────────────┐
│        t₁    t₂    t₃    ...   t₁₀    │
│   t₁ [1.00  0.00  0.00   ...   0.00]   │  (can only see itself)
│   t₂ [0.60  0.40  0.00   ...   0.00]   │  (60% self, 40% prev)
│   t₃ [0.10  0.20  0.70   ...   0.00]   │  (attends to all prior)
│   ...                                  │
│   t₁₀[0.05  0.05  0.05   ...   0.15]   │  (distributed attention)
└────────────────────────────────────────┘
```

**Step 4.5: Weighted Sum (Attention × V)**

```
Attention Weights: (B, n_head, T, T)
V: (B, n_head, T, head_dim)
         ↓
    Attention @ V
         ↓
Output: (B, n_head, T, head_dim)
```

**Visual Diagram:**

```
Attention Weights (single head):
┌─────────────────────────┐
│ [w₁₁ w₁₂ w₁₃ ... w₁₁₀] │  Row 1: how t₁ attends
│ [w₂₁ w₂₂ w₂₃ ... w₂₁₀] │  Row 2: how t₂ attends
│ ...                     │
└─────────────────────────┘

V (single head):
┌──────────────┐
│ v₁: [64 dim] │
│ v₂: [64 dim] │
│ v₃: [64 dim] │
│ ...          │
└──────────────┘
         ↓
    Matrix Multiply
         ↓
Output (single head):
┌──────────────┐
│ o₁ = w₁₁*v₁ + w₁₂*v₂ + ... + w₁₁₀*v₁₀ │
│ o₂ = w₂₁*v₁ + w₂₂*v₂ + ... + w₂₁₀*v₁₀ │
│ ...          │
└──────────────┘
```

**Example calculation for token 2:**
```
o₂ = 0.60*v₁ + 0.40*v₂ + 0.00*v₃ + ... + 0.00*v₁₀
   = weighted combination of value vectors
```

---

#### Step 5: Reshape Back (Line 38)
```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

**Transformation:**

```
Input: (B, n_head, T, head_dim)
         ↓ transpose(1, 2)
Intermediate: (B, T, n_head, head_dim)
         ↓ view(B, T, C)
Output: (B, T, C)  where C = n_head * head_dim
```

**Visual Diagram:**

```
y (before): (B=2, n_head=12, T=10, head_dim=64)
┌────────────────────────────────────────┐
│ Batch 1:                               │
│ Head 1:  [64] [64] ... [64] (10 tokens)│
│ Head 2:  [64] [64] ... [64]            │
│ ...                                    │
│ Head 12: [64] [64] ... [64]            │
└────────────────────────────────────────┘
         ↓ transpose(1, 2)
y (intermediate): (B=2, T=10, n_head=12, head_dim=64)
┌────────────────────────────────────────┐
│ Batch 1:                               │
│ Token 1: [64][64]...[64] (12 heads)   │
│ Token 2: [64][64]...[64]               │
│ ...                                    │
│ Token 10: [64][64]...[64]              │
└────────────────────────────────────────┘
         ↓ view(2, 10, 768)
y (final): (B=2, T=10, C=768)
┌────────────────────────────────────────┐
│ Batch 1:                               │
│ Token 1: [768 dims]                    │
│ Token 2: [768 dims]                    │
│ ...                                    │
│ Token 10: [768 dims]                   │
└────────────────────────────────────────┘
```

**What "concatenation" means:**
- Each head produces a 64-dim output per token
- We concatenate all 12 head outputs → 12×64 = 768 dims
- This combines information from all attention heads

---

#### Step 6: Output Projection (Line 39)
```python
y = self.c_proj(y)
```

**Matrix Operation:**

```
Input: (B, T, C)
         ↓ Linear(c_proj: C → C)
Output: (B, T, C)
```

**Purpose:**
- Applies learned linear transformation
- Allows model to combine information from all heads
- Final dimensionality matches input

---

## Complete Flow Diagram

```
Input x: (B, T, C)
    │
    ├─→ Linear(c_attn) → (B, T, 3C)
    │         │
    │         ├─→ Split → Q: (B, T, C)
    │         ├─→ Split → K: (B, T, C)
    │         └─→ Split → V: (B, T, C)
    │
    ├─→ Reshape → Q: (B, n_head, T, head_dim)
    ├─→ Reshape → K: (B, n_head, T, head_dim)
    └─→ Reshape → V: (B, n_head, T, head_dim)
                │
                ├─→ Q @ K^T → Scores: (B, n_head, T, T)
                │     │
                │     ├─→ Scale (÷√head_dim)
                │     ├─→ Causal Mask
                │     └─→ Softmax → Attention: (B, n_head, T, T)
                │
                └─→ Attention @ V → (B, n_head, T, head_dim)
                            │
                            ├─→ Transpose → (B, T, n_head, head_dim)
                            └─→ View → (B, T, C)
                                    │
                                    └─→ Linear(c_proj) → (B, T, C)
                                                      │
                                                    Output
```

---

## Key Concepts

### Why Multi-Head Attention?
- Different heads learn different patterns:
  - Head 1: Long-range dependencies
  - Head 2: Local patterns
  - Head 3: Syntactic relationships
  - Head 4: Semantic relationships
  - etc.

### Why Causal Masking?
- Prevents information leakage from future tokens
- Essential for autoregressive generation (like GPT)
- Each token can only "see" previous tokens

### What Does Attention Learn?
- Which tokens are relevant to each other
- Contextual relationships in the sequence
- Long-range dependencies that RNNs struggle with

---

## Matrix Multiplication Dimensions Summary

| Operation | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| `c_attn(x)` | (B, T, C) | (B, T, 3C) | Single linear layer |
| Split QKV | (B, T, 3C) | 3×(B, T, C) | Split into 3 tensors |
| Reshape Q/K/V | (B, T, C) | (B, n_head, T, head_dim) | View + transpose |
| Q @ K^T | (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) | (B, n_head, T, T) | Attention scores |
| Attention @ V | (B, n_head, T, T) @ (B, n_head, T, head_dim) | (B, n_head, T, head_dim) | Weighted values |
| Reshape back | (B, n_head, T, head_dim) | (B, T, C) | Concatenate heads |
| `c_proj(y)` | (B, T, C) | (B, T, C) | Output projection |

---

## Example with Real Numbers

Let's trace through with: **B=1, T=5, C=768, n_head=12, head_dim=64**

1. **Input**: `(1, 5, 768)` - one sequence with 5 tokens
2. **QKV projection**: `(1, 5, 2304)` - 768×3 = 2304
3. **Split**: Three tensors of shape `(1, 5, 768)`
4. **Reshape Q/K/V**: `(1, 12, 5, 64)` - 12 heads, 5 tokens, 64 dims each
5. **Attention scores**: `(1, 12, 5, 5)` - each head computes 5×5 attention matrix
6. **Attention output**: `(1, 12, 5, 64)` - each head produces 64-dim output
7. **Concatenate**: `(1, 5, 768)` - all 12 heads concatenated
8. **Output projection**: `(1, 5, 768)` - final output

---

## Notes on Implementation

- **Flash Attention**: `F.scaled_dot_product_attention` uses optimized Flash Attention algorithm
- **Efficiency**: Computing Q, K, V together saves memory and computation
- **Gradient Flow**: The operations are designed to allow efficient backpropagation

