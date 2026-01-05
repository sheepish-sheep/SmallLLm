# How to Run Training - Step-by-Step Guide

This guide shows you exactly what commands to run to start training your GPT model.

---

## Prerequisites Checklist

Before starting, make sure you have:

- ✅ Python installed (3.8+ recommended)
- ✅ PyTorch installed with CUDA support (if using GPU)
- ✅ Required packages installed (see below)
- ✅ Sufficient disk space (dataset is ~10B tokens)
- ✅ GPU with enough memory (or be prepared to use CPU - much slower)

---

## Step 1: Install Required Packages

Install all necessary Python packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tiktoken datasets tqdm requests transformers
```

**Or use a requirements file** (create `requirements.txt`):
```
torch>=2.0.0
numpy
tiktoken
datasets
tqdm
requests
transformers
```

Then install:
```bash
pip install -r requirements.txt
```

---

## Step 2: Prepare Training Data

The training script expects data in the `edu_fineweb10B` directory. You need to download and tokenize the data first.

### Option A: Download FineWeb-Edu Dataset (Recommended)

Run the data preparation script:

```bash
python fineweb.py
```

**What this does:**
- Downloads the FineWeb-Edu dataset from HuggingFace
- Tokenizes all documents using GPT-2 tokenizer
- Saves data shards as `.npy` files in `edu_fineweb10B/` directory
- Creates files like: `train_000.npy`, `train_001.npy`, `val_000.npy`, etc.

**Expected output:**
- Directory: `edu_fineweb10B/`
- Multiple shard files (typically 100+ shards for 10B tokens)
- Each shard contains ~100M tokens

**Note:** This will download ~10-20GB of data and may take some time depending on your internet connection.

### Option B: Use Your Own Data

If you want to use custom data:
1. Create directory: `edu_fineweb10B/`
2. Tokenize your data and save as `.npy` files
3. Name them with pattern: `train_XXX.npy` and `val_XXX.npy`
4. Each file should be a NumPy array of token IDs (uint16)

---

## Step 3: Verify Data Directory

Before training, make sure your data is ready:

```bash
# Check if directory exists
ls edu_fineweb10B/

# Should see files like:
# train_000.npy
# train_001.npy
# train_002.npy
# ...
# val_000.npy
# val_001.npy
```

If the directory doesn't exist or is empty, go back to Step 2.

---

## Step 4: Start Training

Now you can start training! Choose one of the options below based on your setup.

### Option 1: Single GPU / CPU Training (Simplest)

**Command:**
```bash
python train-chunni.py
```

**What happens:**
- Trains on single GPU (if CUDA available) or CPU
- Uses all available GPU memory
- Prints progress to console and saves to `log/log.txt`
- Saves checkpoints every 5000 steps to `log/` directory

**Output example:**
```
using device: cuda
total desired batch size: 524288
=> calculated gradient accumulation steps: 8
num decayed parameter tensors: 145, with 123,456,789 parameters
num non-decayed parameter tensors: 48, with 1,234,567 parameters
using fused AdamW: True
step     0 | loss: 10.823456 | lr 8.3916e-07 | norm: 1.2345 | dt: 125.50ms | tok/sec: 418432.50
validation loss: 10.5432
HellaSwag accuracy: 1234/10042=0.1229
...
```

---

### Option 2: Multi-GPU Training (Faster)

If you have multiple GPUs, you can train much faster using distributed training.

**For 2 GPUs:**
```bash
torchrun --standalone --nproc_per_node=2 train-chunni.py
```

**For 4 GPUs:**
```bash
torchrun --standalone --nproc_per_node=4 train-chunni.py
```

**For 8 GPUs:**
```bash
torchrun --standalone --nproc_per_node=8 train-chunni.py
```

**For N GPUs:**
```bash
torchrun --standalone --nproc_per_node=N train-chunni.py
```

**What happens:**
- Launches N parallel processes (one per GPU)
- Each GPU processes different data
- Gradients are synchronized across GPUs
- Near-linear speedup (8 GPUs ≈ 8x faster)

**Requirements:**
- All GPUs must be on the same machine
- CUDA required (distributed training needs CUDA)
- Sufficient memory on each GPU

---

### Option 3: Training on Multiple Machines (Advanced)

For training across multiple machines:

```bash
# On machine 0 (master):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="IP_OF_MACHINE_0" --master_port=29500 train-chunni.py

# On machine 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="IP_OF_MACHINE_0" --master_port=29500 train-chunni.py
```

**Requirements:**
- High-speed network between machines (InfiniBand recommended)
- Shared filesystem or data on each machine
- Proper firewall/network configuration

---

## Step 5: Monitor Training

Training will run for 19,073 steps (approximately 1 epoch). Monitor progress:

### Console Output

Watch for:
- **Step progress**: Loss decreasing over time
- **Validation loss**: Should decrease (lower is better)
- **HellaSwag accuracy**: Should increase (higher is better)
- **Learning rate**: Changes according to schedule
- **Tokens/sec**: Training speed

### Log Files

Check the log file:
```bash
# View log in real-time
tail -f log/log.txt

# View last 50 lines
tail -n 50 log/log.txt
```

**Log format:**
```
0 train 10.823456
0 val 10.5432
0 hella 0.1229
250 train 8.456789
250 val 8.2345
250 hella 0.1456
...
```

### Checkpoints

Checkpoints are saved to `log/` directory:
```bash
ls log/
# Should see:
# log.txt
# model_05000.pt
# model_10000.pt
# model_15000.pt
# model_19073.pt (final checkpoint)
```

---

## Training Configuration

### Default Settings

The training script uses these defaults (in `train-chunni.py`):

- **Total batch size**: 524,288 tokens (~0.5M)
- **Micro batch size**: 64 sequences
- **Sequence length**: 1024 tokens
- **Model size**: GPT-2 Small (12 layers, 12 heads, 768 dims)
- **Vocab size**: 50,304 tokens
- **Max steps**: 19,073 (~1 epoch)
- **Learning rate**: 6e-4 (with warmup and cosine decay)
- **Weight decay**: 0.1

### Modifying Configuration

To change settings, edit `train-chunni.py`:

**Change batch size:**
```python
total_batch_size = 262144  # Smaller batch (fits in less memory)
B = 32  # Smaller micro batch
```

**Change model size:**
```python
model = GPT(GPTConfig(
    vocab_size=50304,
    n_layer=6,      # Fewer layers (smaller model)
    n_head=6,       # Fewer heads
    n_embd=512      # Smaller embeddings
))
```

**Change training length:**
```python
max_steps = 10000  # Train for fewer steps
```

---

## Expected Training Time

### Single GPU (e.g., RTX 3090)
- **Per step**: ~120ms
- **Total time**: ~38 minutes (no evaluation)
- **With evaluation**: ~2-3 hours

### 8 GPUs (Distributed)
- **Per step**: ~15ms
- **Total time**: ~5 minutes (no evaluation)
- **With evaluation**: ~30-45 minutes

### CPU Only
- **Per step**: ~5-10 seconds
- **Total time**: ~27-53 hours
- **Not recommended** unless you have very powerful CPU

---

## Troubleshooting

### Error: "no shards found for split train"

**Problem:** Data directory is missing or empty.

**Solution:**
```bash
# Check if directory exists
ls edu_fineweb10B/

# If empty or missing, run:
python fineweb.py
```

---

### Error: "CUDA out of memory"

**Problem:** GPU doesn't have enough memory.

**Solutions:**

1. **Reduce batch size:**
   ```python
   total_batch_size = 262144  # Half the default
   B = 32  # Smaller micro batch
   ```

2. **Reduce sequence length:**
   ```python
   T = 512  # Instead of 1024
   ```

3. **Use gradient checkpointing** (requires code modification)

4. **Use CPU** (much slower):
   - The script will auto-detect and use CPU if CUDA unavailable

---

### Error: "for now i think we need CUDA for DDP"

**Problem:** Trying distributed training without CUDA.

**Solution:**
- DDP requires CUDA
- Use single-GPU training: `python train-chunni.py`
- Or use CPU (single process only)

---

### Error: "Cannot forward sequence of length X, block size is only Y"

**Problem:** Input sequence is longer than model's block_size.

**Solution:**
- Reduce sequence length in DataLoader: `T = 512`
- Or increase block_size in GPTConfig: `block_size = 2048`

---

### Training is very slow

**Possible causes:**
1. **Using CPU instead of GPU**
   - Check: `using device: cuda` should appear
   - Install CUDA-enabled PyTorch

2. **Small batch size**
   - GPU not fully utilized
   - Increase batch size if memory allows

3. **Data loading bottleneck**
   - Data on slow disk
   - Move data to SSD if possible

---

## Quick Start Summary

**Fastest way to get started:**

```bash
# 1. Install packages
pip install torch numpy tiktoken datasets tqdm requests transformers

# 2. Download data (takes time - downloads ~10-20GB)
python fineweb.py

# 3. Start training
python train-chunni.py
```

That's it! Training will start automatically.

---

## What to Expect

### Training Progress

You should see:
1. **Loss decreasing**: Starts ~10-11, should drop to ~2-3 by end
2. **Validation loss**: Should track training loss
3. **HellaSwag accuracy**: Starts ~12-13%, should reach ~30-40% by end
4. **Generated text improving**: Check samples every 250 steps

### Output Files

- `log/log.txt`: All training metrics
- `log/model_XXXXX.pt`: Model checkpoints
  - Contains: model weights, config, step number, validation loss

### Using Checkpoints

To load a checkpoint later:
```python
checkpoint = torch.load('log/model_05000.pt')
model.load_state_dict(checkpoint['model'])
```

---

## Advanced Options

### Resume Training from Checkpoint

To resume from a checkpoint, you'll need to modify the code to:
1. Load checkpoint
2. Restore optimizer state
3. Continue from saved step

(Currently not implemented in the basic script)

### Change Dataset

To use different data:
1. Modify `data_root` in DataLoader (line 174)
2. Ensure data format matches (`.npy` files with token arrays)
3. Files must be named with `train_` or `val_` prefix

### Evaluation Only

To evaluate a trained model:
1. Load checkpoint
2. Set model to eval mode
3. Run validation/HellaSwag evaluation only

---

## Next Steps After Training

Once training completes:

1. **Test the model**: Generate text samples
2. **Evaluate**: Run on test sets
3. **Deploy**: Use for inference
4. **Fine-tune**: Continue training on specific tasks

---

## Quick Reference Commands

```bash
# Install dependencies
pip install torch numpy tiktoken datasets tqdm requests transformers

# Prepare data
python fineweb.py

# Single GPU/CPU training
python train-chunni.py

# Multi-GPU training (8 GPUs)
torchrun --standalone --nproc_per_node=8 train-chunni.py

# Monitor training
tail -f log/log.txt

# Check checkpoints
ls -lh log/*.pt
```

---

## Need Help?

If you encounter issues:
1. Check error messages carefully
2. Verify data directory exists and has files
3. Check GPU memory (use `nvidia-smi`)
4. Review log files for detailed error info

Good luck with your training! 🚀

