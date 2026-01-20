# Chuni - VN Style Text-to-Speech

A pipeline that converts any text into **Visual Novel dialogue style** and reads it aloud using voice cloning.

## How to Run

```bash
cd "Basic AI Voice Clone"
python main.py
```

Then:
1. **Highlight any text** on your computer
2. Press `Ctrl+Alt+Shift+V` - rewrites in VN style and speaks it
3. Press `Ctrl+Alt+Shift+O` - toggle overlay window

## How It Works

```
Your Text → Seq2Seq (VN Style) → Voice Cloner → Audio Output

Example:
  Input:  "I would like to inquire about the availability."
  Output: "Hey, is that thing still up for grabs?" [spoken aloud]
```

1. **You highlight text** anywhere on your computer
2. **Seq2Seq model** rewrites it in Visual Novel dialogue style
3. **Voice cloner** generates speech using a cloned voice
4. **Audio plays** through your speakers

## Documentation

| Folder | Description |
|--------|-------------|
| [Architecture](Architecture/) | GPT architecture, self-attention, data loading |
| [Training](Training/) | How to run training, training loop, GPU setup |
| [VN_Pipeline](VN_Pipeline/) | Seq2seq paraphrasing - training, inference, config |
| [VN_Scraping](VN_Scraping/) | Visual novel data scraping tools |
| [VoiceClone](VoiceClone/) | Voice cloning - XTTS, Chatterbox, overlay UI |

## Quick Links

### Getting Started
- [How Training Works](Architecture/HOW_TRAINING_WORKS.md)
- [How to Run Training](Training/HOW_TO_RUN_TRAINING.md)
- [GPU/CUDA Setup](Training/CUDA_PYTORCH_FIX.md)

### VN Pipeline
- [Pipeline Flow](VN_Pipeline/PIPELINE_FLOW.md) - Complete end-to-end guide
- [Generation Explained](VN_Pipeline/GENERATION_EXPLAINED.md) - How text generation works
- [Folder Structure](VN_Pipeline/FOLDER_STRUCTURE.md) - Code organization
- [Config Reference](VN_Pipeline/CONFIG_REFERENCE.md) - All config.json settings

### Voice Cloning
- [How It Works](VoiceClone/HOW_IT_WORKS.md)
- [Improving Quality](VoiceClone/IMPROVING_QUALITY.md)
- [Overlay Guide](VoiceClone/OVERLAY_GUIDE.md)

### Architecture
- [GPT Architecture](Architecture/GPT_ARCHITECTURE_EXPLANATION.md)
- [Self-Attention](Architecture/SELF_ATTENTION_EXPLANATION.md)
- [Encoder-Decoder](VN_Pipeline/ENCODER_DECODER_ARCHITECTURE.md)

## Project Structure

```
Chuni/
├── config.json                      # All settings
├── Basic AI Voice Clone/
│   ├── main.py                      # MAIN APP - run this
│   ├── voice_cloner.py              # Chatterbox voice model
│   ├── voice_cloner_xtts.py         # XTTS-v2 voice model (better)
│   ├── chatterbox/                  # Chatterbox TTS code
│   └── voice_samples/ma/            # Voice reference audio
│
├── EncoderAndMoreInput/
│   ├── VN_Pipeline/                 # Seq2seq paraphrasing
│   │   ├── inference/               # Inference code
│   │   ├── training/                # Training scripts
│   │   └── out/seq2seq/             # Model checkpoints
│   └── encoder_decoder_backup.py    # Model architecture
│
└── docs/                            # Documentation
```

## Voice Models

| Model | Config Value | Quality | Location |
|-------|--------------|---------|----------|
| **XTTS-v2** | `"xtts"` | Better | AppData (downloaded) |
| Chatterbox | `"chatterbox"` | Good | `chatterbox/` folder |

Set in `config.json`:
```json
"voice_model_type": "xtts"
```
