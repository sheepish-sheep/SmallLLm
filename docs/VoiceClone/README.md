# Basic AI Voice Clone

AI voice cloning tool that reads highlighted text in a cloned voice speaking English.

## Quick Start

1. Activate virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Run the app:
   ```powershell
   python main.py
   ```

3. Use it:
   - Highlight text → Copy (Ctrl+C) → Press `Ctrl+Alt+Shift+V` to read
   - Press `Ctrl+Alt+Shift+O` to toggle overlay window

## Requirements

- Python 3.11+
- NVIDIA GPU (optional, but recommended for faster processing)
- Voice samples in `voice_samples/ma/` directory

## Add your own voice samples

Drop a few short, clean clips of the speaker you want to mimic (formats: `.ogg`, `.wav`, `.mp3`, `.flac`) into `voice_samples/ma/`. The app randomly picks from these files when generating speech; 3–5 clips without background noise gives the best results.

## Documentation

See `docs/` folder for detailed documentation.

## Files

- `main.py` - Main application
- `voice_cloner.py` - Voice cloning module
- `overlay_window.py` - Transparent overlay GUI
- `config.py` - Configuration settings
