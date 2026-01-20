# Code Walkthrough (Line-by-Line Commentary)

This document explains the key code paths so you can tweak or extend them later.

## main.py
- Imports: system libs, keyboard/pyperclip/audio libs, project modules (`VoiceCloner`, `config`, `TransparentOverlay`).
- `class VoiceReplicationApp`: container for the service.
  - `__init__`: sets flags, hotkeys, overlay reference, default language (`en`), and expressiveness (0.5).
  - `setup_voice_cloner`: prints status, chooses CUDA if available, builds `VoiceCloner`, creates the overlay, and seeds a “Ready” status.
  - `get_highlighted_text`: tries up to 3 auto-copy attempts via `keyboard.send('ctrl+c')`, waits briefly, returns new clipboard text or falls back to current clipboard.
  - `read_highlighted_text`: fetches text, bails with status if empty, updates overlay status, calls `generate_and_play`, then marks ready.
  - `generate_and_play`: maps expressiveness to `exaggeration/cfg_weight/temperature`, calls `voice_cloner.generate_speech` with current language, plays the WAV via sounddevice, updates overlay status, and cleans up the temp file.
  - `set_language`: stores `"en"` or `"ja"`.
  - `set_expressiveness`: clamps the slider value into 0.0–1.0.
  - `on_keybind_pressed`: launches `read_highlighted_text` in a daemon thread.
  - `on_overlay_toggle`: queues overlay toggle and logs visibility.
  - `start`: marks running, prints hotkey instructions, registers hotkeys, then waits on keyboard (Ctrl+C stops).
  - `stop`: marks not running and logs stop.
- `main()`: instantiates `VoiceReplicationApp`, initializes the cloner, and starts the service.

## overlay_window.py
- Imports tkinter/ttk/threading.
- `class TransparentOverlay`:
  - `__init__`: stores app reference, state flags, pending queues, and UI thread handles; defines character metadata.
  - `create_overlay`: starts the Tk UI thread and waits briefly for readiness.
  - `_run_ui`: builds the Tk root (transparent, always-on-top), positions it, builds widgets, shows the window, schedules periodic updates, and enters `mainloop`.
  - `_build_ui`: creates title, character dropdown, language dropdown (EN/JA), expressiveness slider (0–1), buttons (Read Highlighted, Hide Overlay), and status label.
  - `_schedule_updates`: every 50ms applies pending status/toggle updates on the Tk thread.
  - `on_character_change`: updates the selected character (placeholder hook for future voice switching).
  - `on_language_change`: calls `app.set_language` and posts a status message.
  - `on_expressiveness_change`: calls `app.set_expressiveness` and updates the helper label.
  - `read_highlighted`: posts “Reading…” status and spins off `app.read_highlighted_text`.
  - `show` / `hide`: reveal or withdraw the Tk window and track visibility.
  - `toggle`: queues a visibility toggle; `_apply_pending_toggle` executes it on the Tk thread.
  - `update_status`: queues status text/color; `_apply_pending_status` applies it on the Tk thread.

## voice_cloner.py
- Imports and inserts `chatterbox/src` into `sys.path`, then loads `ChatterboxMultilingualTTS`.
- `class VoiceCloner`:
  - `__init__`: stores voice sample dir and desired device; auto-selects CUDA/MPS/CPU if unset; prints device choice; loads the pre-trained model on that device.
  - `get_voice_prompt_path`: collects audio files in the samples directory, returns a random one (or first) for prompting.
  - `is_ready`: returns whether the model is loaded.
  - `generate_speech`: checks readiness, picks a prompt if needed, calls `self.model.generate` with text, prompt, language, and tunable parameters (exaggeration, cfg_weight, temperature, repetition_penalty), then saves to WAV if an output path is provided.
- `setup_voice_cloning()` helper (for ad-hoc testing) loads the cloner and generates a sample if run directly.

## How to Use/Modify Later
- Change hotkeys in `config.py` (READ_KEYBIND/OVERLAY_KEYBIND).
- Adjust expressiveness mapping in `main.py` (`generate_and_play`).
- Add character-specific prompts by extending `self.characters` and wiring `on_character_change`.
- Change overlay look/position/transparency inside `_run_ui` and `_build_ui`.
- Swap default languages by setting `VoiceReplicationApp.target_language` or changing the overlay default in `_build_ui`.
