# Coding Guide

Use this to understand the current codebase (main.py, overlay_window.py, voice_cloner.py) and how to extend it.

## Flow Overview
1. Hotkeys are configured in `config.py` (`READ_KEYBIND`, `OVERLAY_KEYBIND`).
2. `main.py` initializes `VoiceCloner`, creates the overlay UI, and registers hotkeys.
3. Pressing the read hotkey auto-copies the highlighted text, generates speech, and plays audio.
4. The overlay runs in its own Tk thread so it can toggle visibility and update status without blocking audio.

## Key Files
- `main.py`
  - `VoiceReplicationApp.setup_voice_cloner()` loads the TTS model, choosing GPU if available.
  - `get_highlighted_text()` auto-copies selection (no manual Ctrl+C required).
  - `generate_and_play()` maps the expressiveness slider to exaggeration/CFG/temperature, calls `VoiceCloner.generate_speech()`, and plays the WAV.
  - `set_language()` and `set_expressiveness()` are called from the overlay controls.
  - Hotkeys: read (`READ_KEYBIND`) and overlay toggle (`OVERLAY_KEYBIND`).
- `overlay_window.py`
  - Creates a transparent, always-on-top Tk window in its own thread.
  - Provides language dropdown (EN/JA), expressiveness slider (0.0-1.0), Read Highlighted button, and Hide Overlay button.
  - Queues status/toggle updates and applies them on the Tk thread.
- `voice_cloner.py`
  - Wraps `ChatterboxMultilingualTTS` and picks a random prompt from `voice_samples/ma`.
  - `generate_speech()` exposes exaggeration/CFG/temperature/repetition_penalty for tuning.

## Extending
- Add keybinds: edit `config.py`.
- Add characters: extend `self.characters` in `overlay_window.py` and switch voice prompts in `on_character_change()`.
- Tune audio feel: adjust how expressiveness maps to exaggeration/CFG/temperature in `main.py`.
- Change overlay look: tweak alpha/geometry/fonts in `overlay_window.py`.

## Testing Tips
- `python -m py_compile main.py overlay_window.py` to catch syntax errors quickly.
- Run `python main.py`, press the read hotkey with highlighted text, and verify overlay status changes.
