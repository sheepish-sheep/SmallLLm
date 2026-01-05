"""
Config schema and validation for the VN Pipeline.

This module documents all expected config keys and provides validation.
"""

from pathlib import Path
from typing import Any


# Schema: (key, type, required, default, description)
CONFIG_SCHEMA = [
    # Data paths
    ("vn_cleaned_path", str, True, None, "Path to cleaned VN dialogue text file"),
    ("vn_shard_dir", str, True, None, "Directory containing train_*.npy and val_*.npy shards"),
    ("base_checkpoint_path", str, True, None, "Path to base model checkpoint (e.g., log/model_04999.pt)"),
    
    # Fine-tuning
    ("finetune_output_dir", str, True, None, "Directory to save fine-tuned checkpoints"),
    ("finetune_resume_path", str, False, "", "Resume fine-tuning from this checkpoint"),
    ("finetune_max_steps", int, False, 6000, "Maximum fine-tuning steps"),
    ("finetune_val_steps", int, False, 10, "Validation steps per eval"),
    
    # Seq2seq
    ("seq2seq_output_dir", str, True, None, "Directory for seq2seq outputs (checkpoints, data)"),
    ("seq2seq_checkpoint_path", str, False, None, "Path to seq2seq checkpoint for inference"),
    ("seq2seq_resume_path", str, False, "", "Resume seq2seq training from checkpoint"),
    ("seq2seq_max_steps", int, False, 800, "Maximum seq2seq training steps"),
    ("seq2seq_max_gen_len", int, False, 80, "Maximum generation length"),
    ("seq2seq_len_ratio", float, False, 1.1, "Max output/input length ratio"),
    ("seq2seq_min_gen_len", int, False, 8, "Minimum generation length"),
    ("seq2seq_temperature", float, False, 0.7, "Sampling temperature"),
    ("seq2seq_top_k", int, False, 40, "Top-k sampling"),
    ("seq2seq_span_len", int, False, 3, "Highlight span length (characters)"),
    ("seq2seq_min_line_len", int, False, 15, "Minimum line length for training pairs"),
    ("seq2seq_max_line_len", int, False, 120, "Maximum line length for training pairs"),
    ("seq2seq_source_prefix", str, False, "SOURCE: ", "Prefix added to source text"),
    ("seq2seq_source_suffix", str, False, " => TARGET: ", "Suffix added to source text"),
    ("seq2seq_val_steps", int, False, 10, "Validation steps per eval"),
    ("seq2seq_val_interval", int, False, 50, "Steps between validations"),
    ("seq2seq_save_interval", int, False, 50, "Steps between checkpoint saves"),
    ("seq2seq_learning_rate", float, False, 5e-5, "Seq2seq learning rate"),
    ("seq2seq_force_vn_init", bool, False, True, "Force init from VN checkpoint (ignore resume)"),
    
    # Tokenizer
    ("highlight_start_token", str, False, "<hl>", "Highlight start token"),
    ("highlight_end_token", str, False, "</hl>", "Highlight end token"),
    ("vocab_size", int, False, 50304, "Vocabulary size"),
    
    # Training
    ("max_seq_len", int, False, 512, "Maximum sequence length"),
    ("batch_size", int, False, 8, "Training batch size"),
    ("learning_rate", float, False, 1e-5, "Fine-tuning learning rate"),
    ("num_epochs", int, False, 3, "Number of epochs (unused, max_steps is preferred)"),
    
    # Voice
    ("voice_clone_root", str, False, "Basic AI Voice Clone", "Root directory for voice cloner"),
    ("voice_samples_dir", str, False, None, "Directory with voice sample audio files"),
    ("voice_output_dir", str, False, None, "Directory to save generated audio"),
    ("voice_language", str, False, "en", "Output language code"),
    ("voice_cfg_weight", float, False, 0.35, "CFG/pace weight (0-1)"),
    ("voice_temperature", float, False, 0.8, "Voice generation temperature"),
    ("voice_exaggeration", float, False, 0.5, "Speech expressiveness (0-1)"),
    ("voice_repetition_penalty", float, False, 2.0, "Repetition penalty"),
    ("voice_test_text", str, False, "Hello from the VN pipeline.", "Test text for voice"),
    ("pipeline_voice_texts", list, False, [], "Texts to synthesize in pipeline"),
    ("sample_rate_hz", int, False, 22050, "Audio sample rate"),
]


def validate_config(config: dict, required_only: bool = False) -> list[str]:
    """
    Validate a config dictionary against the schema.
    
    Args:
        config: The config dictionary to validate.
        required_only: If True, only check required fields.
    
    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    
    for key, expected_type, required, default, description in CONFIG_SCHEMA:
        if key not in config:
            if required:
                errors.append(f"Missing required key: '{key}' - {description}")
            continue
        
        value = config[key]
        
        # Skip type check for None values on optional fields
        if value is None and not required:
            continue
        
        # Type checking
        if expected_type == float and isinstance(value, int):
            # Allow int for float fields
            continue
        if not isinstance(value, expected_type):
            errors.append(
                f"Wrong type for '{key}': expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
    
    return errors


def get_config_docs() -> str:
    """
    Generate documentation for all config keys.
    
    Returns:
        Markdown-formatted documentation string.
    """
    lines = ["# Config Keys\n"]
    
    current_section = None
    sections = {
        "vn_": "## Data Paths",
        "base_": "## Data Paths",
        "finetune_": "## Fine-tuning",
        "seq2seq_": "## Seq2Seq Training",
        "highlight_": "## Tokenizer",
        "vocab_": "## Tokenizer",
        "max_seq_": "## Training",
        "batch_": "## Training",
        "learning_": "## Training",
        "num_": "## Training",
        "voice_": "## Voice Synthesis",
        "pipeline_": "## Voice Synthesis",
        "sample_": "## Voice Synthesis",
    }
    
    for key, expected_type, required, default, description in CONFIG_SCHEMA:
        # Determine section
        section = None
        for prefix, sec_name in sections.items():
            if key.startswith(prefix):
                section = sec_name
                break
        if section and section != current_section:
            lines.append(f"\n{section}\n")
            current_section = section
        
        req_str = "**Required**" if required else "Optional"
        default_str = f"Default: `{default}`" if default is not None else ""
        lines.append(f"- `{key}` ({expected_type.__name__}): {description}")
        lines.append(f"  - {req_str}. {default_str}")
    
    return "\n".join(lines)


def apply_defaults(config: dict) -> dict:
    """
    Apply default values to a config dictionary.
    
    Args:
        config: The config dictionary to fill in.
    
    Returns:
        A new dictionary with defaults applied.
    """
    result = dict(config)
    for key, expected_type, required, default, description in CONFIG_SCHEMA:
        if key not in result and default is not None:
            result[key] = default
    return result


if __name__ == "__main__":
    # Print documentation
    print(get_config_docs())

