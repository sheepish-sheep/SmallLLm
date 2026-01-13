# How Voice Cloning Works (No Training Needed)

## Core Idea
- Chatterbox ships as a pre-trained multilingual TTS model.
- Your voice samples are used as prompts, not for training.
- Generation is zero-shot: instant cloning, no fine-tuning step.

## What Happens at Runtime
1. The app auto-copies the highlighted text when you hit the read hotkey.
2. A random voice sample from `voice_samples/ma` is chosen as a prompt.
3. The model extracts tone/timbre from the prompt and generates new speech in your chosen language (EN/JA).
4. Expressiveness, CFG, and temperature are adjusted from the slider.
5. Audio is written to a temp WAV and played immediately.

## Why 862 Samples Help
- More variety = better coverage of tone and prosody.
- Random prompt selection yields natural variation between reads.
- No training loop; all variety comes from your prompt library.

## GPU vs CPU
- If NVIDIA CUDA is available, generation uses GPU for speed.
- Otherwise it falls back to CPU (slower but functional).

## Overlay Thread
- Overlay runs in its own Tk UI thread so it stays responsive.
- Status updates are queued from worker threads (generate/play).
- You can toggle language and expressiveness without restarting.

## Takeaways
- Pre-trained model: nothing to train locally.
- Prompts only guide style; they are not re-learned.
- Hotkey flow is single-press: highlight text, press `Ctrl+Alt+Shift+V`, audio plays.
- Overlay provides visibility (status, language toggle, expressiveness slider).
