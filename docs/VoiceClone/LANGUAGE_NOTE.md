# Language Configuration Note

## Cross-Lingual Setup
- Voice source: Japanese samples in `voice_samples/ma`.
- Output language: user-selectable (English or Japanese) in the overlay dropdown.
- Voice characteristics (timbre/accent) stay Japanese even when speaking English.

## How It Works
1. The app auto-copies highlighted text on the read hotkey.
2. `VoiceCloner.generate_speech()` is called with the current `language` value from the overlay.
3. The same Japanese prompt guides pronunciation and style regardless of output language.

## Defaults
- English (`en`) is the default output.
- Japanese (`ja`) can be selected on demand in the overlay.

## Tips
- For best intelligibility in English, keep expressiveness moderate (slider around 0.4-0.6).
- For a more stylized or robotic feel, lower the slider; for a more animated performance, raise it.

## Where to Change
- Runtime: choose language in the overlay dropdown.
- Code: `VoiceReplicationApp.target_language` and `set_language()` in `main.py`.
- Config: keybinds live in `config.py` (`READ_KEYBIND`, `OVERLAY_KEYBIND`).
