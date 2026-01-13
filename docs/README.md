# Chuni Documentation

All project documentation organized by topic.

## Folders

| Folder | Description |
|--------|-------------|
| [Architecture](Architecture/) | GPT architecture, self-attention, data loading explanations |
| [Training](Training/) | How to run training, training loop details, GPU setup |
| [VN_Pipeline](VN_Pipeline/) | Seq2seq paraphrasing pipeline - training, inference, config |
| [VN_Scraping](VN_Scraping/) | Visual novel data scraping tools and guides |
| [VoiceClone](VoiceClone/) | Voice cloning system - XTTS, Chatterbox, overlay UI |

## Quick Links

### Getting Started
- [How Training Works](Architecture/HOW_TRAINING_WORKS.md)
- [How to Run Training](Training/HOW_TO_RUN_TRAINING.md)
- [GPU/CUDA Setup](Training/CUDA_PYTORCH_FIX.md)

### VN Pipeline (Paraphrasing + Voice)
- [Pipeline Flow](VN_Pipeline/PIPELINE_FLOW.md) - Complete end-to-end guide
- [Generation Explained](VN_Pipeline/GENERATION_EXPLAINED.md) - How text generation works
- [Folder Structure](VN_Pipeline/FOLDER_STRUCTURE.md) - Code organization
- [Config Reference](VN_Pipeline/CONFIG_REFERENCE.md) - All config.json settings

### Voice Cloning
- [How It Works](VoiceClone/HOW_IT_WORKS.md)
- [Improving Quality](VoiceClone/IMPROVING_QUALITY.md)
- [Overlay Guide](VoiceClone/OVERLAY_GUIDE.md)

### Architecture Deep Dives
- [GPT Architecture](Architecture/GPT_ARCHITECTURE_EXPLANATION.md)
- [Self-Attention](Architecture/SELF_ATTENTION_EXPLANATION.md)
- [Encoder-Decoder](VN_Pipeline/ENCODER_DECODER_ARCHITECTURE.md)
