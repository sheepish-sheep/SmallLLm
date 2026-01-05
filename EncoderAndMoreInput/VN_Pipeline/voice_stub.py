"""
Stub for voice model inference.

Goal: take the final text line and synthesize audio using your voice model.
"""

import json
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

_VOICE_MODEL = None


def load_voice_model(voice_clone_root: Path, voice_samples_dir: Path, device: str | None) -> Any:
    """
    Load the Basic AI Voice Clone VoiceCloner class.
    """
    voice_cloner_path = voice_clone_root / "voice_cloner.py"
    if not voice_cloner_path.is_file():
        raise FileNotFoundError(f"voice_cloner.py not found: {voice_cloner_path}")
    spec = importlib.util.spec_from_file_location("voice_cloner", voice_cloner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoiceCloner(voice_samples_dir=str(voice_samples_dir), device=device)


def synthesize(
    model: Any,
    text: str,
    output_path: Path,
    language: str,
    cfg_weight: float,
    temperature: float,
    exaggeration: float,
    repetition_penalty: float,
) -> Path:
    """
    Run TTS synthesis and save audio to output_path.
    """
    model.generate_speech(
        text,
        str(output_path),
        language=language,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        exaggeration=exaggeration,
    )
    return output_path


def speak_text(text: str, config: dict) -> Path:
    """
    Synthesize speech for a text string and return the saved audio path.
    """
    global _VOICE_MODEL
    voice_clone_root = Path(config.get("voice_clone_root", "Basic AI Voice Clone"))
    if not voice_clone_root.is_absolute():
        voice_clone_root = Path(__file__).resolve().parents[2] / voice_clone_root
    voice_samples_dir = Path(config.get("voice_samples_dir", voice_clone_root / "voice_samples" / "ma"))
    if not voice_samples_dir.is_absolute():
        voice_samples_dir = Path(__file__).resolve().parents[2] / voice_samples_dir
    device = config.get("voice_device")
    if _VOICE_MODEL is None:
        _VOICE_MODEL = load_voice_model(voice_clone_root, voice_samples_dir, device)

    output_dir = Path(config.get("voice_output_dir", "EncoderAndMoreInput/VN_Pipeline/out/audio"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"voice_{timestamp}.wav"
    language = config.get("voice_language", "en")
    cfg_weight = float(config.get("voice_cfg_weight", 0.35))
    temperature = float(config.get("voice_temperature", 0.8))
    exaggeration = float(config.get("voice_exaggeration", 0.5))
    repetition_penalty = float(config.get("voice_repetition_penalty", 2.0))
    return synthesize(
        _VOICE_MODEL,
        text,
        output_path,
        language,
        cfg_weight,
        temperature,
        exaggeration,
        repetition_penalty,
    )


def main() -> None:
    """
    Load config.json and generate a test audio clip.
    """
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Config file not found")
    with open(config_path, "r") as f:
        config = json.load(f)
    test_text = config.get("voice_test_text", "Hello from the VN pipeline.")
    output_path = speak_text(test_text, config)
    print(f"Voice output saved to {output_path}")


if __name__ == "__main__":
    main()
