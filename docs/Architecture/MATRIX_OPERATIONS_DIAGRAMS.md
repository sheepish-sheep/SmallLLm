# Matrix Operations Visual Guide - Self-Attention

## Detailed Matrix Multiplication Diagrams

### Example Configuration
- **B** = Batch size = 2
- **T** = Sequence length = 5
- **C** = Embedding dimension = 768
- **n_head** = Number of heads = 12
- **head_dim** = C / n_head = 64

---

## 1. Input to QKV Projection

### Input Tensor: x
```
Shape: (B=2, T=5, C=768)

Batch 1:
┌──────────────────────────────────────────────────────────┐
│ Token 1: [0.1, 0.3, -0.2, ..., 0.5]  (768 dimensions)   │
│ Token 2: [0.2, -0.1, 0.4, ..., 0.3]                      │
│ Token 3: [0.0, 0.5, -0.3, ..., 0.1]                      │
│ Token 4: [-0.2, 0.1, 0.6, ..., -0.2]                     │
│ Token 5: [0.3, -0.4, 0.2, ..., 0.4]                      │
└──────────────────────────────────────────────────────────┘

Batch 2:
┌──────────────────────────────────────────────────────────┐
│ Token 1: [0.4, -0.2, 0.1, ..., 0.2]  (768 dimensions)   │
│ Token 2: [0.1, 0.3, -0.5, ..., 0.5]                      │
│ Token 3: [-0.3, 0.2, 0.4, ..., 0.1]                      │
│ Token 4: [0.2, -0.1, 0.3, ..., -0.3]                     │
│ Token 5: [0.5, 0.1, -0.2, ..., 0.4]                      │
└──────────────────────────────────────────────────────────┘
```

### Linear Layer: c_attn
```
Weight Matrix: (C=768, 3*C=2304)

┌──────────────────────────────────────────┐
│         Output dimensions (2304)         │
│  ┌──┐ ┌──┐ ┌──┐ ... ┌──┐ ┌──┐ ┌──┐    │
│  │w │ │w │ │w │     │w │ │w │ │w │    │
│  └──┘ └──┘ └──┘     └──┘ └──┘ └──┘    │
│ Input 768 dimensions                    │
└──────────────────────────────────────────┘

Operation: x @ W_attn + b_attn
Input: (2, 5, 768) @ (768, 2304) → (2, 5, 2304)
```

### Output: qkv
```
Shape: (B=2, T=5, 3*C=2304)

Batch 1, Token 1:
┌──────────────────────────────────────────────────────────────┐
│ [q₁, q₂, ..., q₇₆₈, k₁, k₂, ..., k₇₆₈, v₁, v₂, ..., v₇₆₈] │
│   ←──────── Q (768) ────────→ ←───── K (768) ───→ ←───── V (768) ───→ │
└──────────────────────────────────────────────────────────────┘
```

### Split Operation
```
qkv: (2, 5, 2304)
         ↓ split(dim=2, size=768)
    ┌────┴────┐
    ↓         ↓         ↓
   Q         K         V
(2,5,768)  (2,5,768)  (2,5,768)

Q = qkv[:, :, 0:768]
K = qkv[:, :, 768:1536]
V = qkv[:, :, 1536:2304]
```

---

## 2. Reshape for Multi-Head Attention

### Before Reshape: Q
```
Shape: (B=2, T=5, C=768)

Batch 1:
┌──────────────────────────────────────────┐
│ Token 1: [q₁, q₂, ..., q₇₆₈] (768 dims) │
│ Token 2: [q₁, q₂, ..., q₇₆₈]            │
│ Token 3: [q₁, q₂, ..., q₇₆₈]            │
│ Token 4: [q₁, q₂, ..., q₇₆₈]            │
│ Token 5: [q₁, q₂, ..., q₇₆₈]            │
└──────────────────────────────────────────┘
```

### After View: Q
```
Shape: (B=2, T=5, n_head=12, head_dim=64)

Batch 1, Token 1:
┌────────────────────────────────────────────┐
│ Head 1:  [q₁, q₂, ..., q₆₄]               │
│ Head 2:  [q₆₅, q₆₆, ..., q₁₂₈]            │
│ Head 3:  [q₁₂₉, q₁₃₀, ..., q₁₉₂]          │
│ ...                                       │
│ Head 12: [q₇₀₅, q₇₀₆, ..., q₇₆₈]         │
└────────────────────────────────────────────┘

Each head gets 64 consecutive dimensions from the 768-dim vector
```

### After Transpose: Q
```
Shape: (B=2, n_head=12, T=5, head_dim=64)

Batch 1, Head 1:
┌────────────────────────────────────────────┐
│ Token 1: [q₁, q₂, ..., q₆₄] (64 dims)     │
│ Token 2: [q₁, q₂, ..., q₆₄]               │
│ Token 3: [q₁, q₂, ..., q₆₄]               │
│ Token 4: [q₁, q₂, ..., q₆₄]               │
│ Token 5: [q₁, q₂, ..., q₆₄]               │
└────────────────────────────────────────────┘

Now organized by head, then tokens, then dimensions
```

---

## 3. Attention Score Computation (Q × K^T)

### Q and K Shapes
```
Q: (B=2, n_head=12, T=5, head_dim=64)
K: (B=2, n_head=12, T=5, head_dim=64)
```

### Matrix Multiplication: Q @ K^T
```
For each head independently:

Q (head 1):                    K^T (head 1):
Shape: (5, 64)                 Shape: (64, 5)
┌─────────────┐                ┌──────────────────────────┐
│ t₁ [64 dim] │                │ k₁₁ k₂₁ k₃₁ k₄₁ k₅₁     │
│ t₂ [64 dim] │                │ k₁₂ k₂₂ k₃₂ k₄₂ k₅₂     │
│ t₃ [64 dim] │      @         │ ...                     │
│ t₄ [64 dim] │                │ k₁₆₄ k₂₆₄ ... k₅₆₄     │
│ t₅ [64 dim] │                └──────────────────────────┘
└─────────────┘

Result: Scores (5, 5)
┌─────────────────────────────────────────────┐
│         t₁      t₂      t₃      t₄      t₅ │
│   t₁ [s₁₁=1.2  s₁₂=0.3  s₁₃=-0.1 s₁₄=0.5  s₁₅=0.8] │
│   t₂ [s₂₁=0.4  s₂₂=1.5  s₂₃=0.2  s₂₄=0.1  s₂₅=0.3] │
│   t₃ [s₃₁=0.1  s₃₂=0.6  s₃₃=1.3  s₃₄=0.4  s₃₅=0.2] │
│   t₄ [s₄₁=0.2  s₄₂=0.1  s₄₃=0.5  s₄₄=1.4  s₄₅=0.6] │
│   t₅ [s₅₁=0.3  s₅₂=0.4  s₅₃=0.2  s₅₄=0.7  s₅₅=1.6] │
└─────────────────────────────────────────────┘

Each s_ij = dot product of query_i and key_j
Example: s₁₂ = q₁ · k₂ = Σ(q₁ₖ × k₂ₖ) for k=1 to 64
```

### Scaling Operation
```
Scores = Scores / √head_dim = Scores / √64 = Scores / 8

After scaling:
┌─────────────────────────────────────────────┐
│         t₁      t₂      t₃      t₄      t₅ │
│   t₁ [0.15   0.04   -0.01   0.06   0.10]   │
│   t₂ [0.05   0.19    0.03   0.01   0.04]   │
│   t₃ [0.01   0.08    0.16   0.05   0.03]   │
│   t₄ [0.03   0.01    0.06   0.18   0.08]   │
│   t₅ [0.04   0.05    0.03   0.09   0.20]   │
└─────────────────────────────────────────────┘
```

### Causal Masking
```
Before masking:                After masking (is_causal=True):
┌─────────────────────┐        ┌─────────────────────┐
│     t₁ t₂ t₃ t₄ t₅ │        │     t₁ t₂ t₃ t₄ t₅ │
│ t₁ [●  ●  ●  ●  ● ]│        │ t₁ [●  ⊗  ⊗  ⊗  ⊗ ]│
│ t₂ [●  ●  ●  ●  ● ]│   →    │ t₂ [●  ●  ⊗  ⊗  ⊗ ]│
│ t₃ [●  ●  ●  ●  ● ]│        │ t₃ [●  ●  ●  ⊗  ⊗ ]│
│ t₄ [●  ●  ●  ●  ● ]│        │ t₄ [●  ●  ●  ●  ⊗ ]│
│ t₅ [●  ●  ●  ●  ● ]│        │ t₅ [●  ●  ●  ●  ● ]│
└─────────────────────┘        └─────────────────────┘

Masked positions set to -inf:
┌─────────────────────────────────────────────┐
│         t₁      t₂      t₃      t₄      t₅ │
│   t₁ [0.15   -inf    -inf    -inf    -inf] │
│   t₂ [0.05   0.19    -inf    -inf    -inf] │
│   t₃ [0.01   0.08    0.16    -inf    -inf] │
│   t₄ [0.03   0.01    0.06    0.18    -inf] │
│   t₅ [0.04   0.05    0.03    0.09    0.20] │
└─────────────────────────────────────────────┘
```

### Softmax Operation
```
Each row → softmax → probabilities (each row sums to 1.0)

After softmax:
┌─────────────────────────────────────────────┐
│         t₁      t₂      t₃      t₄      t₅ │
│   t₁ [1.00   0.00   0.00   0.00   0.00]    │  Sum = 1.0
│   t₂ [0.10   0.90   0.00   0.00   0.00]    │  Sum = 1.0
│   t₃ [0.02   0.05   0.93   0.00   0.00]    │  Sum = 1.0
│   t₄ [0.03   0.01   0.02   0.94   0.00]    │  Sum = 1.0
│   t₅ [0.04   0.05   0.03   0.05   0.83]    │  Sum = 1.0
└─────────────────────────────────────────────┘

Interpretation:
- Token 1: 100% attention on itself (only option)
- Token 2: 10% on t₁, 90% on itself
- Token 3: 2% on t₁, 5% on t₂, 93% on itself
- Token 4: 3% on t₁, 1% on t₂, 2% on t₃, 94% on itself
- Token 5: Distributed attention across all previous tokens
```

---

## 4. Weighted Sum: Attention × V

### Attention Weights and V
```
Attention Weights: (B=2, n_head=12, T=5, T=5)
V: (B=2, n_head=12, T=5, head_dim=64)
```

### Matrix Multiplication: Attention @ V
```
For each head independently:

Attention Weights (head 1):        V (head 1):
Shape: (5, 5)                      Shape: (5, 64)
┌─────────────────────┐            ┌─────────────┐
│ [1.00 0.00 0.00 ...]│            │ t₁ [64 dim] │
│ [0.10 0.90 0.00 ...]│      @     │ t₂ [64 dim] │
│ [0.02 0.05 0.93 ...]│            │ t₃ [64 dim] │
│ [0.03 0.01 0.02 ...]│            │ t₄ [64 dim] │
│ [0.04 0.05 0.03 ...]│            │ t₅ [64 dim] │
└─────────────────────┘            └─────────────┘

Result: (5, 64)
┌────────────────────────────────────────────┐
│ o₁ = 1.00×v₁ + 0.00×v₂ + ... = v₁        │
│ o₂ = 0.10×v₁ + 0.90×v₂ + ...             │
│ o₃ = 0.02×v₁ + 0.05×v₂ + 0.93×v₃ + ...   │
│ o₄ = 0.03×v₁ + 0.01×v₂ + 0.02×v₃ + ...   │
│ o₅ = 0.04×v₁ + 0.05×v₂ + 0.03×v₃ + ...   │
└────────────────────────────────────────────┘
```

### Detailed Calculation Example
```
Output for token 3 (o₃):
o₃ = 0.02×v₁ + 0.05×v₂ + 0.93×v₃ + 0.00×v₄ + 0.00×v₅

Component-wise:
o₃[1] = 0.02×v₁[1] + 0.05×v₂[1] + 0.93×v₃[1] + 0.00×v₄[1] + 0.00×v₅[1]
o₃[2] = 0.02×v₁[2] + 0.05×v₂[2] + 0.93×v₃[2] + 0.00×v₄[2] + 0.00×v₅[2]
...
o₃[64] = 0.02×v₁[64] + 0.05×v₂[64] + 0.93×v₃[64] + 0.00×v₄[64] + 0.00×v₅[64]

Result: 64-dimensional output vector
```

---

## 5. Concatenate All Heads

### Before Concatenation
```
Shape: (B=2, n_head=12, T=5, head_dim=64)

Batch 1, Token 1:
┌────────────────────────────────────────────┐
│ Head 1:  [o₁, o₂, ..., o₆₄]  (64 dims)    │
│ Head 2:  [o₁, o₂, ..., o₆₄]               │
│ Head 3:  [o₁, o₂, ..., o₆₄]               │
│ ...                                       │
│ Head 12: [o₁, o₂, ..., o₆₄]               │
└────────────────────────────────────────────┘
```

### After Transpose and View (Concatenation)
```
Step 1: Transpose (1, 2)
Shape: (B=2, T=5, n_head=12, head_dim=64)

Batch 1:
┌────────────────────────────────────────────┐
│ Token 1:                                   │
│   Head 1:  [64 dims]                       │
│   Head 2:  [64 dims]                       │
│   ...                                      │
│   Head 12: [64 dims]                       │
│                                            │
│ Token 2:                                   │
│   Head 1:  [64 dims]                       │
│   Head 2:  [64 dims]                       │
│   ...                                      │
│   Head 12: [64 dims]                       │
│ ... (5 tokens total)                       │
└────────────────────────────────────────────┘

Step 2: View to concatenate
Shape: (B=2, T=5, C=768)

Batch 1:
┌────────────────────────────────────────────┐
│ Token 1: [h1_64dims || h2_64dims || ... || h12_64dims] │
│          ←────────── 768 dimensions ──────────→        │
│ Token 2: [h1_64dims || h2_64dims || ... || h12_64dims] │
│ ...                                                    │
└────────────────────────────────────────────┘

Each token now has all 12 heads concatenated:
[head1_dim1, ..., head1_dim64, head2_dim1, ..., head2_dim64, ..., head12_dim1, ..., head12_dim64]
```

---

## 6. Output Projection

### Final Linear Layer: c_proj
```
Input: (B=2, T=5, C=768)
         ↓
    Linear(c_proj)
    Weight: (768, 768)
         ↓
Output: (B=2, T=5, C=768)

Weight Matrix:
┌──────────────────────────┐
│      Output (768)        │
│  ┌──┐ ┌──┐ ... ┌──┐    │
│  │w │ │w │     │w │    │
│  └──┘ └──┘     └──┘    │
│ Input (768)             │
└──────────────────────────┘

Operation: y @ W_proj + b_proj
```

---

## Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│                    (B, T, C)                                     │
│                  (2, 5, 768)                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Linear: c_attn       │
        │   (768 → 2304)         │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   QKV: (2, 5, 2304)    │
        └────────┬───────────────┘
                 │
        ┌────────┴────────┐
        │      Split      │
        └─────┬───┬───┬───┘
              │   │   │
              ▼   ▼   ▼
          Q   K   V
    (2,5,768) (2,5,768) (2,5,768)
              │   │   │
        ┌─────┴───┴───┴───┐
        │  Reshape Heads  │
        └─────┬───┬───┬───┘
              │   │   │
              ▼   ▼   ▼
      (2,12,5,64) (2,12,5,64) (2,12,5,64)
              │   │   │
              └───┴───┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   Scaled Dot-Product       │
    │   Attention                │
    │   Q @ K^T / √64            │
    │   → Mask → Softmax         │
    │   → Attention @ V          │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   Output: (2, 12, 5, 64)   │
    └────────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   Transpose & View     │
        │   Concatenate Heads    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   (2, 5, 768)          │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Linear: c_proj       │
        │   (768 → 768)          │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   OUTPUT               │
        │   (B, T, C)            │
        │   (2, 5, 768)          │
        └────────────────────────┘
```

---

## Key Matrix Dimensions Reference

| Operation | Dimensions | Description |
|-----------|-----------|-------------|
| Input | (B, T, C) | Raw input embeddings |
| c_attn | (C, 3C) | Weight matrix for QKV |
| QKV | (B, T, 3C) | Combined Q, K, V |
| Q/K/V | (B, T, C) | Separated queries/keys/values |
| Q/K/V reshaped | (B, n_head, T, head_dim) | Multi-head format |
| Q @ K^T | (B, n_head, T, T) | Attention scores |
| Attention weights | (B, n_head, T, T) | After softmax |
| Attention @ V | (B, n_head, T, head_dim) | Weighted values |
| Concatenated | (B, T, C) | All heads combined |
| c_proj | (C, C) | Output projection weight |
| Output | (B, T, C) | Final result |

---

## Understanding the Flow

1. **Projection**: Input embeddings → Q, K, V representations
2. **Multi-Head Split**: Each head focuses on different aspects
3. **Attention Computation**: Calculate relationships between tokens
4. **Causal Masking**: Prevent future information leakage
5. **Weighted Combination**: Combine value vectors using attention weights
6. **Head Concatenation**: Merge information from all heads
7. **Output Projection**: Final learned transformation

Each step transforms the data in a way that allows the model to learn complex relationships between tokens in the sequence.

