# Complete Training Loop Explanation

This document provides a comprehensive line-by-line explanation of the GPT training loop code (lines 229-473), covering distributed training setup, model initialization, learning rate scheduling, training loop, validation, evaluation, text generation, and checkpointing.

---

## Table of Contents

1. [Setup and Imports](#1-setup-and-imports-lines-229-238)
2. [Distributed Training Setup (DDP)](#2-distributed-training-setup-ddp-lines-240-265)
3. [Device Configuration](#3-device-configuration-lines-267-272)
4. [Token Encoder Setup](#4-token-encoder-setup-line-274)
5. [Batch Size Configuration](#5-batch-size-configuration-lines-276-283)
6. [Data Loader Initialization](#6-data-loader-initialization-lines-285-286)
7. [Model Initialization](#7-model-initialization-lines-288-298)
8. [Learning Rate Schedule](#8-learning-rate-schedule-lines-300-315)
9. [Optimizer Setup](#9-optimizer-setup-line-318)
10. [Logging Setup](#10-logging-setup-lines-320-325)
11. [Main Training Loop](#11-main-training-loop-lines-327-470)
    - [Validation Loss Evaluation](#111-validation-loss-evaluation-lines-332-362)
    - [HellaSwag Evaluation](#112-hellaswag-evaluation-lines-364-395)
    - [Text Generation](#113-text-generation-lines-397-431)
    - [Training Step](#114-training-step-lines-433-469)
12. [Cleanup](#12-cleanup-lines-471-472)

---

## 1. Setup and Imports (Lines 229-238)

### Purpose
Sets up the environment and imports necessary libraries for distributed training.

### Code Breakdown

```python
# -----------------------------------------------------------------------------
# simple launch:
# python train_chunni.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

**Lines 229-233: Comments**
- Line 229: Section separator
- Lines 230-233: Usage instructions
  - Single GPU/CPU: Run directly with Python
  - Multiple GPUs: Use `torchrun` command for distributed training

```python
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
```

**Lines 236-238: Import Distributed Training Modules**
- **Line 236**: Imports functions to initialize and destroy process groups for distributed training
- **Line 237**: Imports `DistributedDataParallel` wrapper for multi-GPU training
- **Line 238**: Imports distributed communication utilities

**What is DDP?**
- **Distributed Data Parallel**: Allows training across multiple GPUs/nodes
- Each GPU processes different data in parallel
- Gradients are synchronized across all processes
- Significantly speeds up training on multi-GPU systems

---

## 2. Distributed Training Setup (DDP) (Lines 240-265)

### Purpose
Detects if running in distributed mode and sets up the appropriate configuration.

### Code Breakdown

```python
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?        
```

**Line 242: Detect DDP Mode**
- Checks for `RANK` environment variable
- `RANK` is set by `torchrun` command when running distributed training
- If `RANK` exists, we're in DDP mode
- `ddp` becomes `True` if distributed, `False` if single-process

```python
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
```

**Lines 243-252: DDP Configuration**

- **Line 245**: Asserts CUDA is available (DDP requires CUDA currently)
- **Line 246**: Initializes process group with NCCL backend (NVIDIA Collective Communications Library)
  - Enables communication between GPUs
  
- **Line 247**: `ddp_rank` - Global rank across all processes (0, 1, 2, ..., N-1)
  - Unique ID for each process across all machines/GPUs
  
- **Line 248**: `ddp_local_rank` - Local rank on this machine (0, 1, 2, ...)
  - Which GPU on this specific machine (0 = first GPU, 1 = second GPU, etc.)
  
- **Line 249**: `ddp_world_size` - Total number of processes (e.g., 8 for 8 GPUs)
  
- **Line 250**: Sets device to specific CUDA device (e.g., `'cuda:0'`, `'cuda:1'`)
  
- **Line 251**: Sets the current CUDA device for this process
  
- **Line 252**: Master process is rank 0 (does logging, checkpointing, etc.)

**Example: 8 GPUs**
```
Process 0: ddp_rank=0, ddp_local_rank=0, device='cuda:0', master=True
Process 1: ddp_rank=1, ddp_local_rank=1, device='cuda:1', master=False
Process 2: ddp_rank=2, ddp_local_rank=2, device='cuda:2', master=False
...
Process 7: ddp_rank=7, ddp_local_rank=7, device='cuda:7', master=False
```

```python
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
```

**Lines 253-265: Single-Process Configuration**

- **Lines 255-258**: Set defaults for single-process run
  - All ranks are 0, world size is 1, always master
  
- **Lines 260-264**: Auto-detect best device
  - Try CUDA first (NVIDIA GPUs)
  - Then MPS (Apple Silicon GPUs)
  - Fallback to CPU
  
- **Line 265**: Print which device is being used

---

## 3. Device Configuration (Lines 267-272)

### Purpose
Configures device type and sets random seeds for reproducibility.

### Code Breakdown

```python
# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"
```

**Line 268: Device Type**
- Extracts device type from device string
- `device_type = "cuda"` if using GPU, `"cpu"` otherwise
- Used for `torch.autocast` (mixed precision training)

```python
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
```

**Lines 270-272: Set Random Seeds**
- **Line 270**: Sets PyTorch random seed to 1337 (for reproducibility)
- **Line 271-272**: Also sets CUDA random seed if GPU is available
- Ensures results are reproducible across runs

**Why 1337?**
- Common seed value in machine learning (references "LEET" from hacker culture)
- Any fixed number works, but this is a convention

---

## 4. Token Encoder Setup (Line 274)

### Purpose
Initializes the tokenizer for encoding/decoding text.

### Code Breakdown

```python
enc = tiktoken.get_encoding("gpt2")
```

**Line 274: Initialize Tokenizer**
- Uses tiktoken library (OpenAI's fast tokenizer)
- Gets GPT-2 encoding (same as GPT-2 uses)
- Used for:
  - Encoding text → token IDs (for generation)
  - Decoding token IDs → text (for displaying generated text)

**What is tiktoken?**
- Fast BPE (Byte Pair Encoding) tokenizer
- Splits text into subword tokens
- Example: "Hello world" → [9906, 1917] (token IDs)

---

## 5. Batch Size Configuration (Lines 276-283)

### Purpose
Configures batch sizes and calculates gradient accumulation steps.

### Code Breakdown

```python
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
```

**Line 276: Total Batch Size**
- `524288` tokens = 2^19 = ~0.5 million tokens
- This is the effective batch size (across all micro-batches and GPUs)

**Line 277: Micro Batch Size**
- `B = 64` sequences per micro-batch
- Limited by GPU memory - can't fit all tokens at once

**Line 278: Sequence Length**
- `T = 1024` tokens per sequence
- Maximum context length the model can handle

**Line 279: Validation**
- Ensures total batch size is divisible by the product
- Prevents rounding errors in gradient accumulation

**Line 280: Calculate Gradient Accumulation Steps**
- Formula: `grad_accum_steps = total_batch_size / (B * T * ddp_world_size)`

**Example Calculation:**

Single GPU:
- `grad_accum_steps = 524288 / (64 * 1024 * 1) = 524288 / 65536 = 8`
- Process 8 micro-batches before updating weights

8 GPUs:
- `grad_accum_steps = 524288 / (64 * 1024 * 8) = 524288 / 524288 = 1`
- Each GPU processes 1 micro-batch, then sync

**Why Gradient Accumulation?**
- Can't fit large batch in GPU memory at once
- Process multiple small batches, accumulate gradients
- Update weights once with accumulated gradients
- Equivalent to large batch training

```python
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
```

**Lines 281-283: Print Configuration**
- Only master process prints (avoids duplicate output)
- Shows total batch size and gradient accumulation steps

---

## 6. Data Loader Initialization (Lines 285-286)

### Purpose
Creates data loaders for training and validation data.

### Code Breakdown

```python
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
```

**Line 285: Training Data Loader**
- Creates DataLoader for training split
- Each process gets different data (based on `ddp_rank`)
- Provides batches of size `B` with sequences of length `T`

**Line 286: Validation Data Loader**
- Creates DataLoader for validation split
- Used to evaluate model performance during training
- Also distributed across processes

**Parameters:**
- `B`: Batch size (64)
- `T`: Sequence length (1024)
- `process_rank`: Which process this is (for data sharding)
- `num_processes`: Total number of processes
- `split`: "train" or "val"

---

## 7. Model Initialization (Lines 288-298)

### Purpose
Creates and configures the GPT model.

### Code Breakdown

```python
torch.set_float32_matmul_precision('high')
```

**Line 288: Set Matrix Multiplication Precision**
- Sets precision for float32 matrix multiplications
- 'high' = faster but slightly less precise
- 'medium' or 'highest' are alternatives

```python
# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
```

**Lines 290-292: Create Model**
- **Line 290**: Creates new GPT model with vocab size 50304
- **Line 291**: Comment shows alternative - load pretrained GPT-2
- **Line 292**: Moves model to device (GPU/CPU/MPS)

```python
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
```

**Lines 293-295: Optional Compilation**
- `torch.compile()` can speed up model (PyTorch 2.0+)
- Currently disabled because it interferes with evaluation
- Would optimize the model graph for faster execution

```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
```

**Lines 296-298: Wrap with DDP and Get Raw Model**
- **Line 297**: Wraps model with DDP wrapper if distributed training
  - Enables gradient synchronization across GPUs
  
- **Line 298**: Gets unwrapped model (removes DDP wrapper)
  - Used for checkpointing (DDP wrapper adds extra state)
  - `model.module` accesses the underlying model when wrapped
  - `model` directly if not wrapped

---

## 8. Learning Rate Schedule (Lines 300-315)

### Purpose
Defines how learning rate changes during training.

### Code Breakdown

```python
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
```

**Lines 300-303: Learning Rate Parameters**
- **Line 300**: Maximum learning rate (0.0006)
- **Line 301**: Minimum learning rate (10% of max = 0.00006)
- **Line 302**: Warmup steps (715) - gradually increase LR
- **Line 303**: Total training steps (19,073) - approximately 1 epoch

**Why these values?**
- Based on GPT-2 training configuration
- 10B tokens dataset, 0.5M tokens/batch = ~20,000 batches
- Adjusted for this specific setup

```python
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
```

**Lines 304-315: Learning Rate Function**

**Phase 1: Linear Warmup (Lines 306-307)**
```python
if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
```
- Gradually increases LR from 0 to max_lr
- Step 0: LR = max_lr * 1/715 ≈ 0.0000008
- Step 715: LR = max_lr * 715/715 = max_lr = 0.0006

**Why Warmup?**
- Prevents early training instability
- Starts with small updates, gradually increases
- Model needs time to adjust to gradients

**Phase 2: After Training (Lines 309-310)**
```python
if it > max_steps:
    return min_lr
```
- If training continues past max_steps, use minimum LR
- Keeps model learning but very slowly

**Phase 3: Cosine Decay (Lines 312-315)**
```python
decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
return min_lr + coeff * (max_lr - min_lr)
```

**How Cosine Decay Works:**
- `decay_ratio` goes from 0 (start) to 1 (end)
- `coeff` uses cosine function:
  - At start (ratio=0): coeff = 0.5 * (1 + cos(0)) = 0.5 * 2 = 1.0
  - At end (ratio=1): coeff = 0.5 * (1 + cos(π)) = 0.5 * 0 = 0.0
- Learning rate smoothly decreases from max_lr to min_lr

**Visual LR Schedule:**

```
Learning Rate
    ↑
max_lr ──────┐
             │     ╱╲
             │    ╱  ╲
             │   ╱    ╲
             │  ╱      ╲
             │ ╱        ╲
             │╱          ╲
min_lr ──────┴──────────────→ Steps
         warmup    cosine decay
```

---

## 9. Optimizer Setup (Line 318)

### Purpose
Creates the optimizer for updating model weights.

### Code Breakdown

```python
# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
```

**Line 318: Create Optimizer**
- Calls `configure_optimizers()` method from GPT model
- Creates AdamW optimizer with:
  - **weight_decay=0.1**: L2 regularization (prevents overfitting)
  - **learning_rate=6e-4**: Initial learning rate (will be overridden by schedule)
  - **device_type**: Used to determine if fused optimizer is available

**What is AdamW?**
- Adaptive Moment Estimation with Weight Decay
- Advanced optimizer that adapts learning rate per parameter
- Better than basic SGD for transformer models

---

## 10. Logging Setup (Lines 320-325)

### Purpose
Creates directory and file for logging training progress.

### Code Breakdown

```python
# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
```

**Line 321: Log Directory**
- Sets directory name to "log"

**Line 322: Create Directory**
- Creates "log" directory if it doesn't exist
- `exist_ok=True` prevents error if directory already exists

**Line 323: Log File Path**
- Creates path to "log/log.txt"

**Lines 324-325: Clear Log File**
- Opens file in write mode (clears existing contents)
- Immediately closes (just clears the file)

**Purpose:**
- All training metrics will be written to this file
- Can track training progress over time
- Useful for plotting loss curves later

---

## 11. Main Training Loop (Lines 327-470)

### Purpose
The core training loop that iterates through training steps, evaluates the model, and updates weights.

### Code Breakdown

```python
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
```

**Line 327: Training Loop**
- Iterates from step 0 to max_steps-1 (19,073 steps)
- Each iteration processes one effective batch

**Line 328: Start Timer**
- Records start time for measuring step duration

**Line 329: Last Step Flag**
- Checks if this is the final step
- Used to ensure evaluation runs on last step

---

## 11.1 Validation Loss Evaluation (Lines 332-362)

### Purpose
Periodically evaluates model on validation data to monitor training progress.

### Code Breakdown

```python
    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
```

**Line 332: Evaluation Condition**
- Runs every 250 steps OR on the last step
- Allows monitoring without slowing training too much

```python
        model.eval()
        val_loader.reset()
```

**Lines 333-334: Prepare for Evaluation**
- **Line 333**: Sets model to evaluation mode
  - Disables dropout, batch norm uses running stats
  - Model behaves differently during eval vs training
  
- **Line 334**: Resets validation loader to start from beginning

```python
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
```

**Lines 335-344: Compute Validation Loss**

- **Line 335**: `torch.no_grad()` context
  - Disables gradient computation (saves memory, faster)
  - Not updating weights, so don't need gradients
  
- **Line 336**: Initialize loss accumulator to 0
  
- **Line 337**: Evaluate on 20 validation batches
  - Average across multiple batches for stability
  
- **Line 338**: Loop through validation batches
  
- **Line 339**: Get next batch from validation loader
  
- **Line 340**: Move data to device (GPU/CPU)
  
- **Line 341**: Mixed precision inference (bfloat16)
  - Faster computation, lower memory
  - Good enough precision for evaluation
  
- **Line 342**: Forward pass through model
  - Gets predictions (logits) and loss
  
- **Line 343**: Divide loss by number of steps
  - Average loss per batch
  
- **Line 344**: Accumulate loss (detached from computation graph)

```python
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
```

**Lines 345-346: Aggregate Across Processes (DDP)**
- If distributed training, average loss across all processes
- All GPUs compute loss on different data
- Average gives overall validation loss

```python
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
```

**Lines 347-350: Log Validation Loss**
- Only master process prints/writes
- Prints to console and writes to log file
- Format: `step val loss_value`

```python
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
```

**Lines 351-362: Save Checkpoint**

- **Line 351**: Save checkpoint every 5000 steps OR on last step
  
- **Line 353**: Create checkpoint filename (e.g., `model_05000.pt`)
  
- **Lines 354-359**: Create checkpoint dictionary
  - `model`: Model weights (state_dict)
  - `config`: Model configuration
  - `step`: Current training step
  - `val_loss`: Validation loss at this step
  
- **Line 362**: Save checkpoint to disk
  - Can resume training from this point later
  - Can use for inference/deployment

**Why Checkpoints?**
- Training can take days/weeks
- If crash occurs, can resume from checkpoint
- Can compare models at different training stages
- Can deploy best model based on validation loss

---

## 11.2 HellaSwag Evaluation (Lines 364-395)

### Purpose
Evaluates model on HellaSwag benchmark (multiple-choice commonsense reasoning).

### Code Breakdown

```python
    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
```

**Line 365: Evaluation Condition**
- Runs every 250 steps (same as validation loss)
- Skips if model is compiled (compilation interferes)

```python
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
```

**Lines 366-371: Distribute Examples Across Processes**

- **Lines 366-367**: Initialize counters for accuracy
  
- **Line 368**: Iterate through HellaSwag validation examples
  
- **Lines 369-371**: Distribute examples across processes
  - Each process evaluates different examples
  - Process 0: examples 0, 8, 16, ...
  - Process 1: examples 1, 9, 17, ...
  - Avoids duplicate work

```python
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
```

**Lines 372-375: Prepare Example**

- **Line 373**: Convert example to tokens and mask
  - `tokens`: All 4 completion options
  - `mask`: Which tokens are part of completion (1) vs prompt (0)
  - `label`: Index of correct answer (0-3)
  
- **Lines 374-375**: Move to device

```python
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
```

**Lines 376-380: Predict Best Completion**

- **Lines 377-379**: Forward pass through model
  - Gets logits for all completions
  
- **Line 380**: Select best completion using `get_most_likely_row()`
  - Evaluates loss for each completion
  - Picks completion with lowest loss

```python
            num_total += 1
            num_correct_norm += int(pred_norm == label)
```

**Lines 381-382: Track Accuracy**
- Increment total examples processed
- Increment correct count if prediction matches label

```python
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
```

**Lines 383-390: Aggregate Statistics (DDP)**
- Sum totals and correct counts across all processes
- Each process evaluated different examples
- Combine to get overall accuracy

```python
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
```

**Lines 391-395: Log Accuracy**
- Calculate accuracy percentage
- Print and log to file

**What is HellaSwag?**
- Benchmark for commonsense reasoning
- Given a story beginning, choose best ending from 4 options
- Tests model's understanding of real-world scenarios
- Good indicator of model quality beyond just loss

---

## 11.3 Text Generation (Lines 397-431)

### Purpose
Periodically generates text from the model to visually inspect training progress.

### Code Breakdown

```python
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
```

**Line 398: Generation Condition**
- Runs every 250 steps (skip step 0 - model is random)
- Skips if compiled (interference)

```python
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
```

**Lines 399-406: Prepare Generation**

- **Line 399**: Set model to eval mode
  
- **Line 400**: Generate 4 different sequences (with different sampling)
  
- **Line 401**: Maximum length of generated text (32 tokens)
  
- **Line 402**: Encode prompt text to token IDs
  
- **Line 403**: Convert to PyTorch tensor
  
- **Line 404**: Create batch of 4 identical prompts
  - `unsqueeze(0)`: Add batch dimension → (1, seq_len)
  - `repeat(4, 1)`: Repeat 4 times → (4, seq_len)
  
- **Line 405**: Move to device

```python
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
```

**Lines 407-408: Random Number Generator**
- Create generator for sampling (ensures reproducibility)
- Seed depends on process rank (different sequences per process)

```python
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
```

**Lines 409-413: Generation Loop**

- **Line 409**: Continue until reaching max length
  
- **Lines 410-413**: Forward pass to get next token predictions
  - Model predicts probability distribution over vocabulary

```python
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
```

**Lines 414-416: Extract Next Token Probabilities**

- **Line 415**: Take logits for last token position only
  - Only need predictions for next token
  
- **Line 416**: Convert logits to probabilities using softmax
  - Sums to 1.0, represents probability distribution

```python
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
```

**Lines 417-420: Top-K Sampling**

- **Line 420**: Get top 50 most likely tokens
  - `topk_probs`: Probabilities of top 50 tokens
  - `topk_indices`: Token IDs of top 50 tokens

**Why Top-K?**
- Prevents sampling very unlikely tokens
- Makes generation more coherent
- Common technique in language models

```python
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
```

**Lines 421-425: Sample Token**

- **Line 423**: Sample 1 token from top-k probabilities
  - Uses multinomial distribution (weighted random choice)
  - More likely tokens have higher chance of being selected
  
- **Line 425**: Get the actual token ID from indices
  - `ix` is index into top-k list
  - `xcol` is the actual token ID

```python
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
```

**Line 427: Append Token**
- Concatenate new token to sequence
- Sequence grows by 1 token each iteration

```python
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
```

**Lines 428-432: Print Generated Text**

- **Line 429**: Loop through all generated sequences
  
- **Line 430**: Convert token IDs to list
  
- **Line 431**: Decode tokens back to text
  
- **Line 432**: Print generated text
  - Shows model's current generation quality
  - Helps visually track training progress

**Example Output:**
```
rank 0 sample 0: Hello, I'm a language model, and I can help you with...
rank 0 sample 1: Hello, I'm a language model, designed to assist users...
rank 0 sample 2: Hello, I'm a language model, trained on vast amounts...
rank 0 sample 3: Hello, I'm a language model, here to answer your...
```

---

## 11.4 Training Step (Lines 433-469)

### Purpose
Performs one training step: forward pass, backward pass, gradient accumulation, and weight update.

### Code Breakdown

```python
    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
```

**Lines 434-437: Prepare Training**

- **Line 434**: Set model to training mode
  - Enables dropout, batch norm uses batch statistics
  
- **Line 435**: Clear gradients from previous step
  - Gradients accumulate, so must clear first
  
- **Line 437**: Initialize loss accumulator

```python
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
```

**Lines 438-440: Gradient Accumulation Loop**

- **Line 438**: Loop through gradient accumulation steps
  - Process multiple micro-batches before updating
  
- **Line 439**: Get next training batch
  
- **Line 440**: Move to device

```python
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
```

**Lines 441-443: DDP Gradient Sync Control**

- **Line 443**: Only sync gradients on last micro-step
  - Intermediate steps: accumulate locally (faster)
  - Final step: sync across all GPUs
  - Optimization for DDP efficiency

```python
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
```

**Lines 444-445: Forward Pass**

- **Line 444**: Mixed precision training (bfloat16)
  - Faster computation, less memory
  - Maintains training stability
  
- **Line 445**: Forward pass through model
  - Gets predictions and loss

```python
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
```

**Lines 446-452: Scale Loss and Backward Pass**

- **Lines 446-449**: Comment explaining why we scale loss
  
- **Line 450**: Divide loss by accumulation steps
  - Gradients add together across micro-batches
  - We want average, not sum
  - Dividing loss before backward achieves this
  
- **Line 451**: Accumulate loss for logging (detached from graph)
  
- **Line 452**: Backward pass - compute gradients
  - Gradients accumulate across micro-batches
  - Only update weights after all micro-batches

**Why Scale Loss?**
```
Without scaling:
  Loss 1: 2.0 → gradients
  Loss 2: 2.0 → gradients
  Loss 3: 2.0 → gradients
  Total gradient = sum of all = 6.0 (too large!)

With scaling:
  Loss 1: 2.0/3 = 0.67 → gradients
  Loss 2: 2.0/3 = 0.67 → gradients
  Loss 3: 2.0/3 = 0.67 → gradients
  Total gradient = sum = 2.0 (correct average!)
```

```python
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
```

**Lines 453-454: Aggregate Loss (DDP)**
- Average loss across all processes
- Each GPU computed loss on different data

```python
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Line 455: Gradient Clipping**
- Clips gradients to maximum norm of 1.0
- Prevents exploding gradients (common in RNNs/Transformers)
- If gradient norm > 1.0, scales down proportionally

**Why Gradient Clipping?**
- Large gradients can cause training instability
- Model weights can explode
- Clipping keeps training stable

```python
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Lines 456-459: Update Learning Rate**

- **Line 457**: Get learning rate for current step (from schedule)
  
- **Lines 458-459**: Update optimizer's learning rate
  - Learning rate changes each step based on schedule

```python
    optimizer.step()
```

**Line 460: Update Weights**
- Performs one optimizer step
- Updates all model parameters using computed gradients
- This is where actual learning happens!

```python
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
```

**Lines 461-462: Synchronize GPU**
- Waits for GPU to finish all operations
- Ensures accurate timing measurements
- GPU operations are asynchronous by default

```python
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
```

**Lines 463-466: Calculate Performance Metrics**

- **Line 463**: Record end time
  
- **Line 464**: Calculate step duration
  
- **Line 465**: Calculate total tokens processed
  - Batch size × sequence length × accumulation steps × world size
  
- **Line 466**: Calculate tokens per second (throughput)

```python
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
```

**Lines 467-470: Log Training Progress**

- **Line 468**: Print comprehensive training info:
  - Step number
  - Loss value
  - Learning rate
  - Gradient norm
  - Step duration (milliseconds)
  - Throughput (tokens/second)
  
- **Line 470**: Write loss to log file

**Example Output:**
```
step     0 | loss: 10.823456 | lr 8.3916e-07 | norm: 1.2345 | dt: 125.50ms | tok/sec: 418432.50
step   250 | loss: 3.456789 | lr 2.0979e-04 | norm: 0.9876 | dt: 118.30ms | tok/sec: 443456.78
```

---

## 12. Cleanup (Lines 471-472)

### Purpose
Cleans up distributed training resources.

### Code Breakdown

```python
if ddp:
    destroy_process_group()
```

**Lines 471-472: Cleanup DDP**
- If using distributed training, properly shut down process group
- Releases communication resources
- Prevents hanging processes

---

## Complete Training Flow Summary

### Single Training Step Flow

```
1. Start Timer
   ↓
2. Validation (every 250 steps)
   - Evaluate on validation data
   - Save checkpoint (every 5000 steps)
   ↓
3. HellaSwag Evaluation (every 250 steps)
   - Test on benchmark
   ↓
4. Text Generation (every 250 steps)
   - Generate sample text
   ↓
5. Training Step
   - Get batch(es) from DataLoader
   - Forward pass (compute loss)
   - Backward pass (compute gradients)
   - Gradient accumulation
   - Gradient clipping
   - Update learning rate
   - Update weights (optimizer.step())
   ↓
6. Log Progress
   - Print metrics
   - Write to log file
   ↓
7. Repeat until max_steps
```

### Key Concepts

1. **Gradient Accumulation**: Process multiple small batches before updating weights
2. **Mixed Precision**: Use bfloat16 for faster training with less memory
3. **Learning Rate Schedule**: Warmup then cosine decay
4. **Distributed Training**: Train across multiple GPUs simultaneously
5. **Evaluation**: Monitor validation loss and benchmark performance
6. **Checkpointing**: Save model periodically for resuming/deployment

---

## Performance Optimizations

1. **Mixed Precision (bfloat16)**: ~2x speedup, ~50% memory reduction
2. **Gradient Accumulation**: Allows large effective batch sizes without large GPU memory
3. **DDP**: Near-linear speedup with multiple GPUs
4. **Top-K Sampling**: Faster than sampling from full vocabulary
5. **Gradient Clipping**: Prevents training instability

---

## Training Time Estimate

**Single GPU:**
- ~19,073 steps
- ~120ms per step
- Total: ~38 minutes (if no evaluation/generation)

**8 GPUs (DDP):**
- Same number of steps
- ~15ms per step (8x faster)
- Total: ~5 minutes

**With Evaluation:**
- Validation adds ~2s every 250 steps
- HellaSwag adds ~30s every 250 steps
- Generation adds ~1s every 250 steps
- Total overhead: ~1-2 hours for full training

---

This completes the comprehensive explanation of the training loop. The code implements a production-ready training setup with distributed training, mixed precision, learning rate scheduling, evaluation, and checkpointing - all essential components for training large language models effectively.

