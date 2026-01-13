# Config Reference

Complete reference for all `config.json` keys used by the VN Pipeline.

## Data Paths

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `vn_cleaned_path` | string | ✓ | Path to cleaned VN dialogue text file |
| `vn_shard_dir` | string | ✓ | Directory for tokenized train/val shards |
| `base_checkpoint_path` | string | ✓ | Path to base GPT checkpoint |

## Fine-tuning

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `finetune_output_dir` | string | required | Directory for fine-tuned checkpoints |
| `finetune_resume_path` | string | `""` | Resume from this checkpoint |
| `finetune_max_steps` | int | `6000` | Maximum fine-tuning steps |
| `finetune_val_steps` | int | `10` | Validation steps per eval |

## Seq2Seq Training

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seq2seq_output_dir` | string | required | Output directory for seq2seq |
| `seq2seq_checkpoint_path` | string | auto | Checkpoint for inference |
| `seq2seq_resume_path` | string | `""` | Resume training from checkpoint |
| `seq2seq_max_steps` | int | `800` | Maximum training steps |
| `seq2seq_learning_rate` | float | `5e-5` | Learning rate |
| `seq2seq_val_steps` | int | `10` | Validation steps per eval |
| `seq2seq_val_interval` | int | `50` | Steps between validations |
| `seq2seq_save_interval` | int | `50` | Steps between saves |
| `seq2seq_force_vn_init` | bool | `true` | Init from VN checkpoint (ignore resume) |

## Seq2Seq Generation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seq2seq_max_gen_len` | int | `80` | Maximum generation length |
| `seq2seq_len_ratio` | float | `1.1` | Max output/input length ratio |
| `seq2seq_min_gen_len` | int | `8` | Minimum generation length |
| `seq2seq_temperature` | float | `0.7` | Sampling temperature |
| `seq2seq_top_k` | int | `40` | Top-k sampling |

## Data Preparation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seq2seq_span_len` | int | `3` | Highlight span length (chars) |
| `seq2seq_min_line_len` | int | `15` | Skip lines shorter than this |
| `seq2seq_max_line_len` | int | `120` | Skip lines longer than this |
| `seq2seq_source_prefix` | string | `"SOURCE: "` | Prefix for source text |
| `seq2seq_source_suffix` | string | `" => TARGET: "` | Suffix for source text |

## Tokenizer

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `highlight_start_token` | string | `"<hl>"` | Highlight start tag |
| `highlight_end_token` | string | `"</hl>"` | Highlight end tag |
| `vocab_size` | int | `50304` | Vocabulary size |

## Training General

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_seq_len` | int | `512` | Maximum sequence length |
| `batch_size` | int | `8` | Training batch size |
| `learning_rate` | float | `1e-5` | Fine-tuning learning rate |
| `shard_size` | int | `1000000` | Tokens per shard file |

## Voice Synthesis

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice_clone_root` | string | `"Basic AI Voice Clone"` | Voice module directory |
| `voice_samples_dir` | string | - | Reference audio samples |
| `voice_output_dir` | string | - | Output audio directory |
| `voice_language` | string | `"en"` | Language code |
| `voice_cfg_weight` | float | `0.35` | CFG/pace (0-1) |
| `voice_temperature` | float | `0.8` | Randomness |
| `voice_exaggeration` | float | `0.5` | Expressiveness (0-1) |
| `voice_repetition_penalty` | float | `2.0` | Repetition penalty |
| `voice_test_text` | string | `"Hello..."` | Test text for voice |
| `pipeline_voice_texts` | list | `[]` | Texts to synthesize |
| `sample_rate_hz` | int | `22050` | Audio sample rate |

## Example config.json

```json
{
  "vn_cleaned_path": "training_data/vn/cleaned_binary_dialogue.txt",
  "base_checkpoint_path": "log/model_04999.pt",
  "vn_shard_dir": "training_data/vn/shards",
  
  "finetune_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs",
  "finetune_max_steps": 6000,
  
  "seq2seq_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/seq2seq",
  "seq2seq_max_steps": 800,
  "seq2seq_learning_rate": 5e-5,
  
  "highlight_start_token": "<hl>",
  "highlight_end_token": "</hl>",
  "max_seq_len": 512,
  "batch_size": 8,
  
  "voice_clone_root": "Basic AI Voice Clone",
  "voice_samples_dir": "Basic AI Voice Clone/voice_samples/ma",
  "voice_output_dir": "EncoderAndMoreInput/VN_Pipeline/out/audio"
}
```

## Validation

Run config validation with:

```python
from EncoderAndMoreInput.VN_Pipeline.utils.config_schema import validate_config
import json

with open("config.json") as f:
    config = json.load(f)

errors = validate_config(config)
if errors:
    for e in errors:
        print(f"Error: {e}")
```

