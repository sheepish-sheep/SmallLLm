# Voice Synthesis Integration

This document explains how the VN Pipeline integrates with the voice cloning module for text-to-speech synthesis.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VOICE SYNTHESIS PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│   │                 │     │                 │     │                 │       │
│   │   Text Input    │────▶│  Seq2SeqRewriter│────▶│  Rewritten Text │       │
│   │                 │     │  (optional)     │     │                 │       │
│   └─────────────────┘     └─────────────────┘     └────────┬────────┘       │
│                                                            │                │
│                                                            ▼                │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                      VOICE CLONING MODULE                       │       │
│   ├─────────────────────────────────────────────────────────────────┤       │
│   │                                                                 │       │
│   │  ┌─────────────────┐     ┌─────────────────┐                    │       │
│   │  │  Reference      │     │   Chatterbox    │                    │       │
│   │  │  Voice Samples  │────▶│   TTS Model     │                    │       │
│   │  │  (.wav/.ogg)    │     │                 │                    │       │
│   │  └─────────────────┘     └────────┬────────┘                    │       │
│   │                                   │                             │       │
│   │                                   ▼                             │       │
│   │  ┌─────────────────────────────────────────────────────────┐    │       │
│   │  │                    SYNTHESIS STEPS                      │    │       │
│   │  │                                                         │    │       │
│   │  │  1. Load reference audio                                │    │       │
│   │  │  2. Extract voice characteristics (embeddings)          │    │       │
│   │  │  3. Process input text                                  │    │       │
│   │  │  4. Generate mel spectrogram                            │    │       │
│   │  │  5. Convert to waveform (vocoder)                       │    │       │
│   │  │  6. Save as .wav file                                   │    │       │
│   │  │                                                         │    │       │
│   │  └─────────────────────────────────────────────────────────┘    │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│                                   │                                         │
│                                   ▼                                         │
│                          ┌─────────────────┐                                │
│                          │   Audio Output  │                                │
│                          │    (.wav file)  │                                │
│                          └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Voice Cloning Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VOICE CLONING PROCESS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: Load Reference Voice                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   voice_samples/                                                │       │
│   │   └── ma/                                                       │       │
│   │       ├── sample1.wav  ──┐                                      │       │
│   │       ├── sample2.ogg  ──┼──▶ Load & concatenate                │       │
│   │       └── sample3.mp3  ──┘                                      │       │
│   │                                                                 │       │
│   │   Requirements:                                                 │       │
│   │   • 5-30 seconds of clean speech per sample                     │       │
│   │   • Multiple samples improve consistency                        │       │
│   │   • Avoid background noise                                      │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   STEP 2: Extract Voice Embeddings                                          │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   Reference Audio ───▶ Voice Encoder ───▶ Voice Embedding       │       │
│   │                                                                 │       │
│   │   The embedding captures:                                       │       │
│   │   • Pitch characteristics                                       │       │
│   │   • Speaking pace                                               │       │
│   │   • Tone quality                                                │       │
│   │   • Accent features                                             │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   STEP 3: Text Processing                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   "Hello from the VN pipeline!"                                 │       │
│   │            │                                                    │       │
│   │            ▼                                                    │       │
│   │   ┌─────────────────┐                                           │       │
│   │   │   Tokenizer     │ → Phoneme conversion                      │       │
│   │   └────────┬────────┘                                           │       │
│   │            │                                                    │       │
│   │            ▼                                                    │       │
│   │   [HH, AH, L, OW, ...]  (phoneme sequence)                      │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   STEP 4: Generate Audio                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   Phonemes + Voice Embedding                                    │       │
│   │            │                                                    │       │
│   │            ▼                                                    │       │
│   │   ┌─────────────────┐                                           │       │
│   │   │  Mel Generator  │  (acoustic model)                         │       │
│   │   └────────┬────────┘                                           │       │
│   │            │                                                    │       │
│   │            ▼                                                    │       │
│   │   Mel Spectrogram (frequency over time)                         │       │
│   │            │                                                    │       │
│   │            ▼                                                    │       │
│   │   ┌─────────────────┐                                           │       │
│   │   │    Vocoder      │  (waveform generator)                     │       │
│   │   └────────┬────────┘                                           │       │
│   │            │                                                    │       │
│   │            ▼                                                    │       │
│   │   Audio Waveform (.wav)                                         │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration Parameters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VOICE CONFIGURATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    QUALITY CONTROLS                             │       │
│   ├─────────────────────────────────────────────────────────────────┤       │
│   │                                                                 │       │
│   │   temperature (0.5 - 1.5)                                       │       │
│   │   ├── 0.5: More consistent, robotic                             │       │
│   │   ├── 0.8: Natural, some variation (DEFAULT)                    │       │
│   │   └── 1.5: Very expressive, may be unstable                     │       │
│   │                                                                 │       │
│   │   cfg_weight (0.0 - 1.0)                                        │       │
│   │   ├── 0.0: No guidance, faster                                  │       │
│   │   ├── 0.35: Balanced (DEFAULT)                                  │       │
│   │   └── 1.0: Maximum guidance, slower                             │       │
│   │                                                                 │       │
│   │   exaggeration (0.0 - 1.0)                                      │       │
│   │   ├── 0.0: Flat, monotone                                       │       │
│   │   ├── 0.5: Natural expression (DEFAULT)                         │       │
│   │   └── 1.0: Very dramatic                                        │       │
│   │                                                                 │       │
│   │   repetition_penalty (1.0 - 2.5)                                │       │
│   │   ├── 1.0: No penalty                                           │       │
│   │   ├── 2.0: Moderate prevention (DEFAULT)                        │       │
│   │   └── 2.5: Strong prevention                                    │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    PATH SETTINGS                                │       │
│   ├─────────────────────────────────────────────────────────────────┤       │
│   │                                                                 │       │
│   │   voice_clone_root: "Basic AI Voice Clone"                      │       │
│   │   └── Root directory of voice module                            │       │
│   │                                                                 │       │
│   │   voice_samples_dir: "Basic AI Voice Clone/voice_samples/ma"    │       │
│   │   └── Directory containing reference audio files                │       │
│   │                                                                 │       │
│   │   voice_output_dir: "EncoderAndMoreInput/VN_Pipeline/out/audio" │       │
│   │   └── Where to save generated audio files                       │       │
│   │                                                                 │       │
│   │   voice_language: "en"                                          │       │
│   │   └── Language code (en, es, fr, de, it, pt, pl, zh, ja, ko)   │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Device Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE ACCELERATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   The voice model automatically selects the best device:                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   Check CUDA (NVIDIA GPU)                                       │       │
│   │        │                                                        │       │
│   │        ├── Yes ───▶ Use "cuda" ───▶ FASTEST (~1-2 sec/sentence) │       │
│   │        │                                                        │       │
│   │        └── No                                                   │       │
│   │             │                                                   │       │
│   │             ▼                                                   │       │
│   │   Check MPS (Apple Silicon)                                     │       │
│   │        │                                                        │       │
│   │        ├── Yes ───▶ Use "mps" ───▶ FAST (~3-5 sec/sentence)     │       │
│   │        │                                                        │       │
│   │        └── No                                                   │       │
│   │             │                                                   │       │
│   │             ▼                                                   │       │
│   │   Use "cpu" ───▶ SLOW (~10-30 sec/sentence)                     │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   Override in config.json:                                                  │
│   {                                                                         │
│       "voice_device": "cuda"  // or "mps" or "cpu"                          │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Full Pipeline Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE VN PIPELINE WITH VOICE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   1. RAW VN TEXT                                                │       │
│   │      "The protagonist entered the <hl>mysterious</hl> room."    │       │
│   │                                                                 │       │
│   └────────────────────────────┬────────────────────────────────────┘       │
│                                │                                            │
│                                ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   2. SEQ2SEQ REWRITER                                           │       │
│   │      Input:  "The protagonist entered the <hl>mysterious</hl>   │       │
│   │               room."                                            │       │
│   │      Output: "The protagonist entered the dark room."           │       │
│   │                                                                 │       │
│   └────────────────────────────┬────────────────────────────────────┘       │
│                                │                                            │
│                                ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   3. VOICE SYNTHESIS                                            │       │
│   │      Input:  "The protagonist entered the dark room."           │       │
│   │      Output: audio_001.wav                                      │       │
│   │                                                                 │       │
│   └────────────────────────────┬────────────────────────────────────┘       │
│                                │                                            │
│                                ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   4. FINAL OUTPUT                                               │       │
│   │      • Rewritten text file                                      │       │
│   │      • Audio narration (.wav)                                   │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### From Pipeline

```python
# In config.json
{
    "pipeline_voice_texts": ["Hello world", "Testing voice"],
    "voice_clone_root": "Basic AI Voice Clone",
    "voice_samples_dir": "Basic AI Voice Clone/voice_samples/ma"
}
```

### Standalone Voice

```python
from EncoderAndMoreInput.VN_Pipeline.voice_stub import speak_text
import json

with open("config.json") as f:
    config = json.load(f)

audio_path = speak_text("Hello from the VN pipeline!", config)
print(f"Audio saved to: {audio_path}")
```

### Direct VoiceCloner

```python
from pathlib import Path
import sys
sys.path.insert(0, "Basic AI Voice Clone")
from voice_cloner import VoiceCloner

cloner = VoiceCloner(
    voice_samples_dir="Basic AI Voice Clone/voice_samples/ma",
    device="cuda"  # or "mps" or "cpu"
)

cloner.generate_speech(
    text="Hello world",
    output_audio_path="output.wav",
    language="en",
    cfg_weight=0.35,
    temperature=0.8,
    exaggeration=0.5
)
```

## Quality Presets

### Natural Speech

```json
{
    "voice_cfg_weight": 0.3,
    "voice_temperature": 1.0,
    "voice_exaggeration": 0.4
}
```

### Consistent/Controlled Speech

```json
{
    "voice_cfg_weight": 0.5,
    "voice_temperature": 0.7,
    "voice_exaggeration": 0.5
}
```

### Dramatic/Expressive

```json
{
    "voice_cfg_weight": 0.4,
    "voice_temperature": 0.9,
    "voice_exaggeration": 0.8
}
```

## File Locations

| File | Purpose |
|------|---------|
| `VN_Pipeline/voice_stub.py` | Integration layer for the pipeline |
| `Basic AI Voice Clone/voice_cloner.py` | Core voice cloning class |
| `Basic AI Voice Clone/chatterbox/` | Chatterbox TTS library |
| `Basic AI Voice Clone/voice_samples/` | Reference voice audio files |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Audio sounds robotic | Increase temperature (0.9-1.1) |
| Audio has artifacts | Decrease temperature (0.6-0.7) |
| Words get repeated | Increase repetition_penalty (2.0-2.5) |
| Output too fast/slow | Adjust cfg_weight |
| Wrong pronunciation | Try different language setting |
| Out of memory | Use smaller batch or CPU |
