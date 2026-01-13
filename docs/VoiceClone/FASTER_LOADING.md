# Making Model Loading Faster

## Why It Takes Time
- Large pre-trained model (~3GB).
- First run downloads; later runs load from cache.
- CPU is slower; GPU (CUDA) is much faster when available.

## Current Optimizations
- Model loads once on startup and stays in memory.
- GPU is auto-detected; falls back to CPU if needed.
- Cached weights are reused after the first download.
- Overlay runs in its own UI thread, so the UI never blocks loading.

## Tips
- Keep the app running; restarting forces a reload.
- Install the CUDA build of PyTorch if you have an NVIDIA GPU (see INSTALL_GPU_PYTORCH.md).
- Avoid shrinking the model; quality would drop.

## Expected Times (approx.)
- First run: download + load (2–3 minutes total, depending on bandwidth/CPU).
- Later runs: load from cache (~10–20s on GPU, ~30–60s on CPU).
