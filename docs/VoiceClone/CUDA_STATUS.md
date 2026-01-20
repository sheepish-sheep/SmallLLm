# CUDA Status

## GPU Snapshot
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- CUDA: 12.9
- Driver: 577.03
- Memory: ~6 GB

## What the App Does
- On startup, `main.py` checks `torch.cuda.is_available()` and picks `"cuda"` when possible.
- `voice_cloner.py` loads `ChatterboxMultilingualTTS` on the chosen device and prints the GPU name.
- If CUDA is missing, it falls back to CPU automatically.

## Verify CUDA
```powershell
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device name:", torch.cuda.get_device_name(0))
PY
```

## Expected Startup Logs
- NVIDIA GPU detected
- Using GPU for faster processing
- Initializing on device: cuda

## If CUDA Is Not Used
1. Check `python -c "import torch; print(torch.cuda.is_available())"`.
2. Ensure `nvidia-smi` reports your GPU.
3. Reinstall CUDA PyTorch (example for CU121):
   ```powershell
   pip uninstall -y torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
