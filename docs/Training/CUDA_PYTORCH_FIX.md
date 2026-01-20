# Fixing PyTorch CUDA Runtime Error

## Current Setup
- **PyTorch Version**: `2.7.1+cu118` (CUDA 11.8)
- **Python Version**: 3.14.0
- **Error**: `torch_python.dll` cannot be found or one of its dependencies is missing

---

## Solution 1: Install CUDA Runtime 11.8 (Recommended)

Your PyTorch version `2.7.1+cu118` requires **CUDA 11.8 runtime libraries**.

### Step 1: Download CUDA 11.8 Runtime

1. **Go to NVIDIA CUDA Downloads:**
   - Visit: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Or search: "CUDA Toolkit 11.8 download"

2. **Select Windows:**
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **11.8**
   - Installer Type: **exe (local)** or **exe (network)**

3. **Download and Install:**
   - Download the installer (usually ~3GB)
   - Run the installer
   - Choose "Express Installation" (recommended)
   - This installs CUDA runtime libraries needed by PyTorch

### Step 2: Install Visual C++ Redistributables

PyTorch also needs Visual C++ Redistributables:

1. **Download Visual C++ Redistributables:**
   - Visit: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
   - Download: **Microsoft Visual C++ Redistributable for Visual Studio 2015-2022**
   - Get the **x64** version

2. **Install it:**
   - Run the installer
   - Restart your computer if prompted

### Step 3: Verify Installation

After installing, restart your terminal and test:

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Test PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Solution 2: Use Older PyTorch Version (If CUDA 11.8 doesn't work)

If you want to try an older, more stable PyTorch version:

### Option A: PyTorch 2.0.1 (CUDA 11.8)
```powershell
.\venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\venv\Scripts\python.exe -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Option B: PyTorch 2.1.0 (CUDA 11.8)
```powershell
.\venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\venv\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Option C: PyTorch 2.2.0 (CUDA 11.8)
```powershell
.\venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\venv\Scripts\python.exe -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## Solution 3: Use CPU-Only PyTorch (No CUDA needed)

If you don't have a GPU or want to avoid CUDA issues:

```powershell
.\venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\venv\Scripts\python.exe -m pip install torch torchvision torchaudio
```

**Note:** Training will be **much slower** on CPU, but it will work.

---

## Solution 4: Use CUDA 12.1 (Newer CUDA)

If you want to use a newer CUDA version:

1. **Install CUDA 12.1:**
   - Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive

2. **Install PyTorch with CUDA 12.1:**
```powershell
.\venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Reference: PyTorch Versions

| PyTorch Version | CUDA Version | Install Command |
|----------------|--------------|----------------|
| 2.7.1 (current) | 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| 2.2.0 | 11.8 | `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118` |
| 2.1.0 | 11.8 | `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118` |
| 2.0.1 | 11.8 | `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118` |
| Latest | 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| Latest | CPU | `pip install torch torchvision torchaudio` |

---

## Recommended Steps (In Order)

1. ✅ **First, try installing CUDA 11.8 Runtime** (Solution 1)
2. ✅ **Install Visual C++ Redistributables** (Solution 1, Step 2)
3. ✅ **Restart your computer**
4. ✅ **Test PyTorch import**
5. ❌ If still doesn't work, try **older PyTorch version** (Solution 2)
6. ❌ If still doesn't work, try **CPU-only version** (Solution 3)

---

## Check Your GPU

To see if you have an NVIDIA GPU:

```powershell
nvidia-smi
```

If this command works, you have an NVIDIA GPU and can use CUDA.
If it doesn't work, you might not have an NVIDIA GPU, so use CPU-only PyTorch.

---

## After Fixing

Once PyTorch works, verify everything:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available()); print('✅ CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

Then run your training:
```powershell
python train-chunni.py
```

