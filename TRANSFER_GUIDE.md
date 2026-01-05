# Transfer Guide: Moving Training to Another Computer

## What to Transfer

### ✅ MUST Transfer (Essential Files):
1. **Code Files:**
   - `train-chunni.py` (main training script)
   - `fineweb.py` (data download script)
   - `hellaswag.py` (evaluation helper)
   - `requirements.txt` (package dependencies)

2. **Data:**
   - `edu_fineweb10B/` (all 100 .npy files - THIS IS LARGE!)
   - `hellaswag/hellaswag_val.jsonl` (evaluation data)

3. **Checkpoints (if you want to resume):**
   - `log/` folder (contains training logs and checkpoints)

4. **Optional (Documentation):**
   - All the explanation folders if you want to keep them

### ❌ DON'T Transfer (Recreate on New Computer):
- `venv/` folder (too large, recreate it)
- `__pycache__/` (Python cache, not needed)

---

## Transfer Methods

### Option 1: External Hard Drive/USB
- Copy the entire folder EXCEPT `venv/` and `__pycache__/`
- The `edu_fineweb10B/` folder is large (~several GB) so make sure you have space

### Option 2: Network Transfer
- Use a network share or cloud storage
- Again, skip `venv/` and `__pycache__/`

### Option 3: Cloud Storage (OneDrive/Google Drive)
- You're already using OneDrive! Just sync the folder
- Skip `venv/` and `__pycache__/` (add to .gitignore if using git)

---

## Setup on New Computer

### 1. Transfer Files
Copy all files (except `venv/` and `__pycache__/`) to the new computer.

### 2. Create Virtual Environment
```powershell
# Navigate to project folder
cd SmallLLm

# Create new virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies
```powershell
# Make sure venv is activated (you should see (venv) in prompt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tiktoken datasets tqdm requests transformers
```

### 4. Verify Data
Make sure `edu_fineweb10B/` folder has all 100 .npy files:
```powershell
(Get-ChildItem edu_fineweb10B\*.npy).Count
# Should output: 100
```

---

## Resume Training

### Option A: Continue from Checkpoint (if you saved one)
```python
# In train-chunni.py, add code to load checkpoint:
# checkpoint = torch.load('log/model_1000.pt')
# model.load_state_dict(checkpoint['model'])
# step = checkpoint['step']
```

### Option B: Start Fresh
```powershell
# Just run:
python train-chunni.py
```

---

## System Requirements for New Computer

- **GPU**: CUDA-compatible GPU (NVIDIA) recommended
  - Check: `nvidia-smi` should work
- **CUDA**: Version 11.8 or 12.1 (for PyTorch)
- **Python**: 3.8 or higher
- **RAM**: 16GB+ recommended
- **Storage**: Enough space for:
  - `edu_fineweb10B/` folder (~several GB)
  - Training checkpoints (~500MB each)
  - Virtual environment (~2-3GB)

---

## Quick Checklist

- [ ] Transfer `train-chunni.py`
- [ ] Transfer `fineweb.py`
- [ ] Transfer `hellaswag.py`
- [ ] Transfer `requirements.txt`
- [ ] Transfer `edu_fineweb10B/` folder (all 100 files)
- [ ] Transfer `hellaswag/` folder
- [ ] Transfer `log/` folder (if resuming)
- [ ] Skip `venv/` (recreate on new computer)
- [ ] Skip `__pycache__/` (not needed)
- [ ] Create venv on new computer
- [ ] Install packages from requirements.txt
- [ ] Verify GPU works: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Run training: `python train-chunni.py`

---

## Troubleshooting

### "CUDA not available"
- Install CUDA toolkit for your GPU
- Reinstall PyTorch with CUDA support

### "No module named X"
- Make sure venv is activated
- Reinstall: `pip install -r requirements.txt`

### "No shards found"
- Verify `edu_fineweb10B/` folder exists and has .npy files
- Check path in `train-chunni.py` (should be `"edu_fineweb10B"`)

