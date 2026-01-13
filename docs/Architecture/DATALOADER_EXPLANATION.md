# DataLoader and Evaluation Functions Explained

This document provides a detailed line-by-line explanation of the DataLoader class and evaluation functions (lines 155-227) in the GPT training code.

---

## Table of Contents

1. [loadTokens Function](#1-loadtokens-function-lines-159-163)
2. [DataLoader Class Overview](#2-dataloader-class-overview)
3. [DataLoader.__init__ Method](#3-dataloader__init__-method-lines-166-183)
4. [DataLoader.reset Method](#4-dataloaderreset-method-lines-185-189)
5. [DataLoader.next_batch Method](#5-dataloadernext_batch-method-lines-191-203)
6. [get_most_likely_row Function](#6-get_most_likely_row-function-lines-209-226)

---

## 1. loadTokens Function (Lines 159-163)

### Purpose
Loads token data from a NumPy file and converts it to a PyTorch tensor for use in training.

### Code Breakdown

```python
def loadTokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.from_numpy(npt).long()
    return ptt
```

### Line-by-Line Explanation

**Line 159: Function Definition**
```python
def loadTokens(filename):
```
- Defines a function named `loadTokens` that takes a `filename` parameter
- This function will load token data from disk

**Line 160: Load NumPy Array**
```python
    npt = np.load(filename)
```
- Uses NumPy's `load()` function to read a `.npy` file
- `.npy` files are NumPy's binary format for storing arrays
- Stores the loaded array in variable `npt` (NumPy tokens)
- The file contains a 1D array of token IDs (integers)

**Line 161: Convert to int32**
```python
    npt = npt.astype(np.int32)
```
- Converts the array to 32-bit integers using `astype()`
- Ensures consistent data type (int32) for all token IDs
- This is important for memory efficiency and type consistency

**Line 162: Convert to PyTorch Tensor**
```python
    ptt = torch.from_numpy(npt).long()
```
- Converts the NumPy array to a PyTorch tensor using `torch.from_numpy()`
- `.long()` converts to PyTorch's long integer type (64-bit)
- Stores result in `ptt` (PyTorch tokens)
- PyTorch tensors are needed for GPU computation and automatic differentiation

**Line 163: Return Tensor**
```python
    return ptt
```
- Returns the PyTorch tensor containing all token IDs
- The tensor is a 1D array of integers representing tokenized text

### Visual Flow

```
File: tokens.npy (on disk)
    ↓
np.load(filename)
    ↓
NumPy Array: [1, 2, 3, 4, 5, ...] (int64 or float)
    ↓
astype(np.int32)
    ↓
NumPy Array: [1, 2, 3, 4, 5, ...] (int32)
    ↓
torch.from_numpy().long()
    ↓
PyTorch Tensor: tensor([1, 2, 3, 4, 5, ...]) (torch.long)
    ↓
Return to caller
```

### Example Usage

```python
# Load tokens from a file
tokens = loadTokens("train_000.npy")
print(tokens.shape)  # torch.Size([1000000]) - 1 million tokens
print(tokens[:10])   # tensor([101, 234, 567, 890, 123, ...])
```

---

## 2. DataLoader Class Overview

### Purpose
The `DataLoader` class manages loading training data from multiple shard files. It handles:
- Loading data from multiple files (shards)
- Creating batches of input/target pairs for training
- Distributed training support (multiple processes)
- Automatic shard cycling when one runs out

### Key Concepts

**Shards**: Large datasets are split into multiple files called "shards" for easier handling
- Example: `train_000.npy`, `train_001.npy`, `train_002.npy`, etc.

**Batching**: Groups tokens into batches for efficient training
- Batch size (B): Number of sequences processed together
- Sequence length (T): Number of tokens per sequence

**Distributed Training**: Multiple processes train on different parts of data simultaneously

---

## 3. DataLoader.__init__ Method (Lines 166-183)

### Purpose
Initializes the DataLoader with configuration and finds all data shard files.

### Code Breakdown

```python
def __init__(self, B, T, process_rank, num_processes, split):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in {'train', 'val'}
    
    # get the shard filenames
    data_root = "edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards
    assert len(shards) > 0, f"no shards found for split {split}"
    if master_process:
        print(f"found {len(shards)} shards for split {split}")
    self.reset()
```

### Line-by-Line Explanation

**Line 166: Constructor Definition**
```python
def __init__(self, B, T, process_rank, num_processes, split):
```
- `B`: Batch size (e.g., 4, 8, 16) - number of sequences per batch
- `T`: Sequence length (e.g., 512, 1024) - tokens per sequence
- `process_rank`: Process ID in distributed training (0, 1, 2, ...)
- `num_processes`: Total number of training processes
- `split`: Either 'train' or 'val' (validation)

**Line 167: Store Batch Size**
```python
    self.B = B
```
- Saves batch size as instance variable

**Line 168: Store Sequence Length**
```python
    self.T = T
```
- Saves sequence length as instance variable

**Line 169: Store Process Rank**
```python
    self.process_rank = process_rank
```
- Saves which process this DataLoader belongs to
- Used to ensure different processes get different data

**Line 170: Store Number of Processes**
```python
    self.num_processes = num_processes
```
- Saves total number of processes
- Used to calculate data offsets for distributed training

**Line 171: Validate Split**
```python
    assert split in {'train', 'val'}
```
- Ensures split is either 'train' or 'val'
- Raises error if invalid value provided

**Line 174: Set Data Directory**
```python
    data_root = "edu_fineweb10B"
```
- Sets the root directory containing data shard files
- This is where all the `.npy` token files are stored

**Line 175: List All Files**
```python
    shards = os.listdir(data_root)
```
- Lists all files and directories in the data root
- Example result: `['train_000.npy', 'train_001.npy', 'val_000.npy', 'other_file.txt']`

**Line 176: Filter by Split**
```python
    shards = [s for s in shards if split in s]
```
- Filters to only files containing 'train' or 'val' in filename
- If `split='train'`: keeps only files with 'train' in name
- If `split='val'`: keeps only files with 'val' in name

**Line 177: Sort Filenames**
```python
    shards = sorted(shards)
```
- Sorts filenames alphabetically
- Ensures consistent order: `train_000.npy`, `train_001.npy`, `train_002.npy`, etc.

**Line 178: Create Full Paths**
```python
    shards = [os.path.join(data_root, s) for s in shards]
```
- Combines directory path with filename
- Converts: `'train_000.npy'` → `'edu_fineweb10B/train_000.npy'`

**Line 179: Store Shard List**
```python
    self.shards = shards
```
- Saves the list of full paths to all shard files

**Line 180: Verify Shards Found**
```python
    assert len(shards) > 0, f"no shards found for split {split}"
```
- Ensures at least one shard file was found
- Raises error with helpful message if none found

**Line 181-182: Print Shard Count**
```python
    if master_process:
        print(f"found {len(shards)} shards for split {split}")
```
- Only the master process (rank 0) prints to avoid duplicate messages
- Prints how many shard files were found

**Line 183: Initialize State**
```python
    self.reset()
```
- Calls `reset()` method to initialize loading state
- Sets current shard to 0 and loads first batch of tokens

### Example Initialization

```python
# Create DataLoader for training with 4 processes
loader = DataLoader(
    B=4,                # batch size of 4
    T=1024,             # sequences of 1024 tokens
    process_rank=0,     # this is process 0 (master)
    num_processes=4,    # 4 processes total
    split='train'       # training data
)

# Result:
# - Finds all train_*.npy files
# - Sorts them: train_000.npy, train_001.npy, ...
# - Loads first shard
# - Sets starting position based on process rank
```

---

## 4. DataLoader.reset Method (Lines 185-189)

### Purpose
Resets the DataLoader to start from the beginning of the data.

### Code Breakdown

```python
def reset(self):
    # state, init at shard zero
    self.current_shard = 0
    self.tokens = loadTokens(self.shards[self.current_shard])
    self.current_position = self.B * self.T * self.process_rank
```

### Line-by-Line Explanation

**Line 185: Method Definition**
```python
def reset(self):
```
- Defines method to reset DataLoader state
- Called during initialization and can be called manually

**Line 187: Reset Shard Index**
```python
    self.current_shard = 0
```
- Sets current shard to 0 (first shard)
- Tracks which shard file we're currently reading from

**Line 188: Load First Shard**
```python
    self.tokens = loadTokens(self.shards[self.current_shard])
```
- Loads tokens from the first shard file using `loadTokens()` function
- Stores entire shard in memory as PyTorch tensor
- Example: If shards[0] = "edu_fineweb10B/train_000.npy", loads that file

**Line 189: Set Starting Position**
```python
    self.current_position = self.B * self.T * self.process_rank
```
- Sets the starting position in the token array based on process rank
- **Why?** For distributed training, each process should start at a different position

### Distributed Training Position Calculation

**Example: B=4, T=1024, 4 processes**

- **Process 0**: position = 4 × 1024 × 0 = 0 (starts at beginning)
- **Process 1**: position = 4 × 1024 × 1 = 4096 (starts 4096 tokens later)
- **Process 2**: position = 4 × 1024 × 2 = 8192 (starts 8192 tokens later)
- **Process 3**: position = 4 × 1024 × 3 = 12288 (starts 12288 tokens later)

This ensures each process gets different data and they don't overlap!

### Visual Example

```
Shard file tokens: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ...]
                     │   │   │   │   │   │   │   │   │   │
Process 0 starts ────┘   │   │   │   │   │   │   │   │   │
Process 1 starts ────────────┘   │   │   │   │   │   │   │
Process 2 starts ────────────────────┘   │   │   │   │   │
Process 3 starts ────────────────────────────┘   │   │   │
```

Each process reads different parts of the data simultaneously.

---

## 5. DataLoader.next_batch Method (Lines 191-203)

### Purpose
Retrieves the next batch of training data (input sequences and target sequences).

### Code Breakdown

```python
def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T) # inputs
    y = (buf[1:]).view(B, T) # targets
    # advance the position in the tensor
    self.current_position += B * T * self.num_processes
    # if loading the next batch would be out of bounds, advance to next shard
    if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = loadTokens(self.shards[self.current_shard])
        self.current_position = B * T * self.process_rank
    return x, y
```

### Line-by-Line Explanation

**Line 191: Method Definition**
```python
def next_batch(self):
```
- Defines method to get the next batch of training data
- Called repeatedly during training loop

**Line 192: Shorten Variable Names**
```python
    B, T = self.B, self.T
```
- Creates local variables for convenience
- `B` = batch size, `T` = sequence length

**Line 193: Extract Token Chunk**
```python
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
```
- Extracts a contiguous chunk of tokens from current position
- Length: `B*T+1` (we need +1 extra token for creating targets)
- Example: If B=2, T=3, extracts 7 tokens: `[t0, t1, t2, t3, t4, t5, t6]`

**Line 194: Create Input Sequences**
```python
    x = (buf[:-1]).view(B, T) # inputs
```
- Takes all but last token: `buf[:-1]` = `[t0, t1, t2, t3, t4, t5]`
- Reshapes to `(B, T)`: 
  ```
  [[t0, t1, t2],
   [t3, t4, t5]]
  ```
- These are the input sequences fed to the model

**Line 195: Create Target Sequences**
```python
    y = (buf[1:]).view(B, T) # targets
```
- Takes all but first token: `buf[1:]` = `[t1, t2, t3, t4, t5, t6]`
- Reshapes to `(B, T)`:
  ```
  [[t1, t2, t3],
   [t4, t5, t6]]
  ```
- These are what the model should predict (inputs shifted by 1 position)

### Why Shift by 1?

In language modeling, we predict the **next token** given previous tokens:

```
Input (x):  [The, cat, sat]
Target (y): [cat, sat, on]
```

For each position, model predicts: "given 'The cat sat', predict 'on'"

### Visual Example

**Example: B=2, T=4**

```
Original tokens: [10, 20, 30, 40, 50, 60, 70, 80, 90]
                     │
                     └─ current_position

Extract buf (B*T+1 = 9 tokens):
buf = [10, 20, 30, 40, 50, 60, 70, 80, 90]

Create inputs (buf[:-1]):
[10, 20, 30, 40, 50, 60, 70, 80]
    ↓ reshape to (2, 4)
x = [[10, 20, 30, 40],
     [50, 60, 70, 80]]

Create targets (buf[1:]):
[20, 30, 40, 50, 60, 70, 80, 90]
    ↓ reshape to (2, 4)
y = [[20, 30, 40, 50],
     [60, 70, 80, 90]]
```

**Line 197: Advance Position**
```python
    self.current_position += B * T * self.num_processes
```
- Moves position forward for next batch
- Jumps by `B * T * num_processes` to account for all processes reading data
- Example: If B=4, T=1024, num_processes=4, jumps by 16,384 tokens

**Line 199: Check if Shard Exhausted**
```python
    if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
```
- Checks if next batch would go beyond current shard's length
- If yes, need to load next shard

**Line 200: Cycle to Next Shard**
```python
        self.current_shard = (self.current_shard + 1) % len(self.shards)
```
- Moves to next shard index
- `% len(self.shards)` wraps around (if last shard, goes back to first)
- Example: If 10 shards, shard 9 → shard 0

**Line 201: Load New Shard**
```python
        self.tokens = loadTokens(self.shards[self.current_shard])
```
- Loads tokens from the new shard file into memory
- Replaces old shard with new one

**Line 202: Reset Position**
```python
        self.current_position = B * T * self.process_rank
```
- Resets position based on process rank (same as in `reset()`)
- Each process starts at different position in new shard

**Line 203: Return Batch**
```python
    return x, y
```
- Returns input sequences `x` and target sequences `y`
- Shapes: `x` = `(B, T)`, `y` = `(B, T)`

### Complete Flow Diagram

```
Current State:
  - current_shard = 0
  - current_position = 8192
  - tokens = [t0, t1, ..., t1000000] (loaded shard)

1. Extract chunk:
   buf = tokens[8192:8192+13] = [t8192, t8193, ..., t8204]

2. Create inputs:
   x = buf[:-1].view(2, 6) = [[t8192...t8197], [t8198...t8203]]

3. Create targets:
   y = buf[1:].view(2, 6) = [[t8193...t8198], [t8199...t8204]]

4. Advance position:
   current_position = 8192 + 24 = 8216

5. Check bounds:
   if 8216 + 13 > 1000000? No, continue...

6. Return (x, y)
```

### Shard Cycling Example

```
Shards: [train_000.npy, train_001.npy, train_002.npy]

Process:
1. Load train_000.npy → read batches → reach end
2. Switch to train_001.npy → read batches → reach end
3. Switch to train_002.npy → read batches → reach end
4. Wrap around → train_000.npy (infinite cycling)
```

---

## 6. get_most_likely_row Function (Lines 209-226)

### Purpose
Helper function for HellaSwag evaluation that selects the best completion from multiple choices based on model loss.

### Context: HellaSwag Evaluation

HellaSwag is a multiple-choice question answering task:
- Given a prompt (start of story)
- 4 possible completions (endings)
- Model must pick the most likely/correct completion

This function evaluates all 4 completions and picks the one with lowest loss.

### Code Breakdown

```python
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
```

### Input Parameters

- **tokens**: `(num_completions, sequence_length)` - Token IDs for all 4 completions
- **mask**: `(num_completions, sequence_length)` - Binary mask (1 = completion region, 0 = prompt)
- **logits**: `(num_completions, sequence_length, vocab_size)` - Model predictions for each completion

### Line-by-Line Explanation

**Line 209: Function Definition**
```python
def get_most_likely_row(tokens, mask, logits):
```
- Takes tokens, mask, and logits for multiple completions
- Returns index (0-3) of the best completion

**Line 211: Shift Logits**
```python
    shift_logits = (logits[..., :-1, :]).contiguous()
```
- Removes last position from logits
- Shape: `(4, seq_len-1, vocab_size)`
- Why? In autoregressive models, we predict next token, so logits at position `i` predict token at position `i+1`

**Line 212: Shift Tokens**
```python
    shift_tokens = (tokens[..., 1:]).contiguous()
```
- Removes first token from tokens (shifts by 1)
- Shape: `(4, seq_len-1)`
- Aligns tokens with their corresponding predictions

**Line 213-214: Flatten for Loss Calculation**
```python
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
```
- Flattens logits: `(4, seq_len-1, vocab_size)` → `(4*(seq_len-1), vocab_size)`
- Flattens tokens: `(4, seq_len-1)` → `(4*(seq_len-1),)`
- Needed for cross-entropy loss function

**Line 215: Compute Per-Token Losses**
```python
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
```
- Computes cross-entropy loss for each token position
- `reduction='none'` means keep individual losses (don't average)
- Shape: `(4*(seq_len-1),)` - one loss value per token

**Line 216: Reshape Losses**
```python
    shift_losses = shift_losses.view(tokens.size(0), -1)
```
- Reshapes back to: `(num_completions, seq_len-1)`
- Each row = losses for one completion

**Line 218: Shift Mask**
```python
    shift_mask = (mask[..., 1:]).contiguous()
```
- Shifts mask by 1 (same as tokens)
- Aligns mask with loss positions
- Shape: `(4, seq_len-1)`

**Line 219: Mask Losses**
```python
    masked_shift_losses = shift_losses * shift_mask
```
- Multiplies losses by mask (0 or 1)
- Zeros out losses in prompt region (where mask=0)
- Keeps only completion region losses (where mask=1)

**Line 221: Sum Losses Per Completion**
```python
    sum_loss = masked_shift_losses.sum(dim=1)
```
- Sums losses across sequence dimension
- Result: one total loss per completion
- Shape: `(4,)` - one loss value per completion

**Line 222: Average Loss Per Completion**
```python
    avg_loss = sum_loss / shift_mask.sum(dim=1)
```
- Divides sum by number of completion tokens (mask positions = 1)
- Gets average loss per token in completion region
- Shape: `(4,)` - average loss for each completion

**Line 225: Find Best Completion**
```python
    pred_norm = avg_loss.argmin().item()
```
- Finds index of completion with lowest average loss
- `argmin()` returns index (0, 1, 2, or 3)
- `.item()` converts tensor to Python int

**Line 226: Return Index**
```python
    return pred_norm
```
- Returns index of the most likely completion (0-3)

### Visual Example

**Input:**
```
tokens (4 completions, 10 tokens each):
  Row 0: [prompt tokens... | completion tokens...]
  Row 1: [prompt tokens... | completion tokens...]
  Row 2: [prompt tokens... | completion tokens...]
  Row 3: [prompt tokens... | completion tokens...]

mask (1 = completion, 0 = prompt):
  Row 0: [0, 0, 0, 0, 0, | 1, 1, 1, 1, 1]
  Row 1: [0, 0, 0, 0, 0, | 1, 1, 1, 1, 1]
  Row 2: [0, 0, 0, 0, 0, | 1, 1, 1, 1, 1]
  Row 3: [0, 0, 0, 0, 0, | 1, 1, 1, 1, 1]

logits (predictions for each token):
  Shape: (4, 10, vocab_size)
```

**Processing:**
```
1. Shift logits and tokens (align for prediction)
2. Compute loss at each position
3. Mask out prompt region losses
4. Average loss in completion region only
5. Find completion with lowest average loss
```

**Output:**
```
pred_norm = 2  (completion 2 has lowest loss = most likely)
```

### Why This Works

- **Lower loss** = model is more confident/accurate
- **Completion region only** = only evaluate the part we care about
- **Average per token** = fair comparison even if completions have different lengths

---

## Complete Data Flow Summary

### Training Data Flow

```
Data Shards (files on disk)
    ↓
loadTokens() - Load shard into memory
    ↓
DataLoader.next_batch() - Extract batch
    ↓
Create (input, target) pairs
    ↓
Feed to GPT model for training
```

### Evaluation Data Flow

```
4 Completions + Mask + Model Logits
    ↓
get_most_likely_row()
    ↓
Compute loss for each completion
    ↓
Mask to completion region only
    ↓
Average loss per completion
    ↓
Return index of best completion
```

---

## Key Concepts Recap

### DataLoader Concepts

1. **Sharding**: Split large datasets into multiple files
2. **Batching**: Group sequences for parallel processing
3. **Distributed Training**: Multiple processes handle different data
4. **Input/Target Shift**: Targets are inputs shifted by 1 position (next token prediction)

### Evaluation Concepts

1. **Autoregressive Prediction**: Model predicts next token at each position
2. **Loss Computation**: Cross-entropy measures prediction quality
3. **Masking**: Focus evaluation on specific regions (completions)
4. **Multiple Choice**: Compare multiple options and pick best

---

## Common Use Cases

### Training Loop Usage

```python
loader = DataLoader(B=8, T=1024, process_rank=0, num_processes=1, split='train')

for epoch in range(num_epochs):
    loader.reset()  # Start from beginning
    for batch_idx in range(num_batches):
        x, y = loader.next_batch()  # Get batch
        logits, loss = model(x, target=y)  # Train
        # ... training code ...
```

### Evaluation Usage

```python
# Evaluate HellaSwag example
tokens = torch.tensor([[prompt + completion0], 
                       [prompt + completion1],
                       [prompt + completion2],
                       [prompt + completion3]])
mask = torch.tensor([[0...0, 1...1],  # prompt=0, completion=1
                     [0...0, 1...1],
                     [0...0, 1...1],
                     [0...0, 1...1]])

logits = model(tokens)  # Get predictions
best_idx = get_most_likely_row(tokens, mask, logits)
print(f"Best completion: {best_idx}")
```

---

This completes the explanation of the DataLoader and evaluation functions. These components are essential for efficiently loading training data and evaluating model performance on multiple-choice tasks.

