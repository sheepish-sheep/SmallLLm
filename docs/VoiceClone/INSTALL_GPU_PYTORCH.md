# Installing GPU Version of PyTorch (NVIDIA)

## Current Status
- CPU build works but is slower.
- Install the CUDA build to speed up generation and loading.

## Check for NVIDIA GPU
```powershell
nvidia-smi
```
If your GPU appears, proceed with CUDA PyTorch.

## Install CUDA PyTorch
```powershell
# Remove CPU builds
pip uninstall -y torch torchaudio

# Install CUDA 12.1 build (most common)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# If you need CUDA 11.8 instead:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verify
```powershell
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

## After Installing
- Run `python main.py`; it should log that CUDA/GPU is detected.
- No code changes required—`main.py` and `voice_cloner.py` auto-select GPU when available.
- Overlay and hotkeys work the same regardless of device.
