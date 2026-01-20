# Basic AI Voice Clone

AI voice cloning tool that reads highlighted text in a cloned voice, with VN-style paraphrasing.

## Quick Start

1. Activate virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Run the app:
   ```powershell
   cd "Basic AI Voice Clone"
   python main.py
   ```

3. Use it:
   - Highlight text anywhere
   - Press `Ctrl+Alt+Shift+V` to rewrite in VN style and speak
   - Press `Ctrl+Alt+Shift+O` to toggle overlay window

## Voice Models

Two voice models are available:

| Model | Config Value | Quality | Notes |
|-------|--------------|---------|-------|
| **XTTS-v2** | `"xtts"` | Better, more natural | Downloads to AppData (~1.8GB) |
| Chatterbox | `"chatterbox"` | Good | Local `chatterbox/` folder |

Set in `config.json`:
```json
"voice_model_type": "xtts"
```

## Requirements

- Python 3.11+
- NVIDIA GPU (recommended for speed)
- Voice samples in `voice_samples/ma/` directory

## Add Your Own Voice

Drop short, clean audio clips into `voice_samples/ma/`:
- Formats: `.ogg`, `.wav`, `.mp3`, `.flac`
- 3-5 clips without background noise works best
- The app randomly picks from these for variety

## Files

| File | Purpose |
|------|---------|
| `main.py` | Main application - background keybind listener |
| `voice_cloner.py` | Chatterbox TTS voice cloner |
| `voice_cloner_xtts.py` | XTTS-v2 voice cloner (better quality) |
| `overlay_window.py` | Transparent overlay GUI |
| `config.py` | Local config (keybinds, paths) |
| `chatterbox/` | Chatterbox TTS model code |

## Documentation

- [How It Works](HOW_IT_WORKS.md)
- [Improving Quality](IMPROVING_QUALITY.md)
- [Overlay Guide](OVERLAY_GUIDE.md)
