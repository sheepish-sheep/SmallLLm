"""
Unified inference: seq2seq paraphrasing + voice synthesis.

This module chains the seq2seq rewriter with voice synthesis to:
1. Take input text (formal/stiff phrasing)
2. Paraphrase it using the seq2seq model (casual VN style)
3. Synthesize speech from the paraphrased text

Usage:
    # As module
    from EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak import rewrite_and_speak
    audio_path = rewrite_and_speak("I'm not entirely convinced of your claim.")

    # As CLI
    python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak "Your text here"
"""

import json
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import Seq2SeqRewriter

# Lazy-loaded singletons
_REWRITER: Optional[Seq2SeqRewriter] = None
_CONFIG: Optional[dict] = None


def get_config() -> dict:
    """Load and cache config."""
    global _CONFIG
    if _CONFIG is None:
        config_path = repo_root / "config.json"
        _CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
    return _CONFIG


def get_rewriter() -> Seq2SeqRewriter:
    """Load and cache seq2seq rewriter."""
    global _REWRITER
    if _REWRITER is None:
        config_path = repo_root / "config.json"
        _REWRITER = Seq2SeqRewriter(config_path, repo_root)
    return _REWRITER


def rewrite_text(text: str) -> str:
    """
    Rewrite text using the seq2seq paraphrasing model.

    Args:
        text: Input text to paraphrase

    Returns:
        Paraphrased text in casual VN style
    """
    rewriter = get_rewriter()
    return rewriter.rewrite(text)


def speak_text(text: str) -> Path:
    """
    Synthesize speech from text.

    Args:
        text: Text to speak

    Returns:
        Path to generated audio file
    """
    from EncoderAndMoreInput.VN_Pipeline import voice_stub
    config = get_config()
    return voice_stub.speak_text(text, config)


def rewrite_and_speak(
    text: str,
    skip_rewrite: bool = False,
    skip_voice: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: paraphrase text and synthesize speech.

    Args:
        text: Input text to process
        skip_rewrite: If True, skip seq2seq paraphrasing
        skip_voice: If True, skip voice synthesis (just return rewritten text)
        verbose: If True, print progress

    Returns:
        dict with keys:
            - input: Original input text
            - rewritten: Paraphrased text (or input if skip_rewrite)
            - audio_path: Path to audio file (or None if skip_voice)
    """
    result = {
        "input": text,
        "rewritten": text,
        "audio_path": None,
    }

    # Step 1: Paraphrase
    if not skip_rewrite:
        if verbose:
            print(f"[Rewrite] Input: {text}")
        rewritten = rewrite_text(text)
        result["rewritten"] = rewritten
        if verbose:
            print(f"[Rewrite] Output: {rewritten}")

    # Step 2: Voice synthesis
    if not skip_voice:
        if verbose:
            print(f"[Voice] Synthesizing: {result['rewritten']}")
        audio_path = speak_text(result["rewritten"])
        result["audio_path"] = audio_path
        if verbose:
            print(f"[Voice] Saved to: {audio_path}")

    return result


def batch_rewrite_and_speak(
    texts: list[str],
    skip_rewrite: bool = False,
    skip_voice: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Process multiple texts through the pipeline.

    Args:
        texts: List of input texts
        skip_rewrite: If True, skip seq2seq paraphrasing
        skip_voice: If True, skip voice synthesis
        verbose: If True, print progress

    Returns:
        List of result dicts (same format as rewrite_and_speak)
    """
    results = []
    for i, text in enumerate(texts, 1):
        if verbose:
            print(f"\n=== Processing {i}/{len(texts)} ===")
        result = rewrite_and_speak(text, skip_rewrite, skip_voice, verbose)
        results.append(result)
    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Paraphrase text and synthesize speech"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to process. If not provided, runs interactive mode."
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Skip paraphrasing, just synthesize voice"
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Skip voice synthesis, just show paraphrased text"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (enter texts one by one)"
    )

    args = parser.parse_args()

    if args.interactive or args.text is None:
        print("Interactive mode. Enter text to paraphrase and speak.")
        print("Commands: 'quit' to exit, 'rewrite-only' to skip voice, 'voice-only' to skip rewrite")
        print("-" * 60)

        skip_rewrite = args.no_rewrite
        skip_voice = args.no_voice

        while True:
            try:
                text = input("\nEnter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not text:
                continue
            if text.lower() == "quit":
                break
            if text.lower() == "rewrite-only":
                skip_voice = True
                skip_rewrite = False
                print("Mode: rewrite only (no voice)")
                continue
            if text.lower() == "voice-only":
                skip_rewrite = True
                skip_voice = False
                print("Mode: voice only (no rewrite)")
                continue
            if text.lower() == "full":
                skip_rewrite = False
                skip_voice = False
                print("Mode: full pipeline (rewrite + voice)")
                continue

            result = rewrite_and_speak(text, skip_rewrite, skip_voice)
            print(f"\nResult: {result['rewritten']}")
            if result["audio_path"]:
                print(f"Audio: {result['audio_path']}")
    else:
        result = rewrite_and_speak(
            args.text,
            skip_rewrite=args.no_rewrite,
            skip_voice=args.no_voice,
        )
        if not args.no_voice and result["audio_path"]:
            print(f"\nDone! Audio saved to: {result['audio_path']}")


if __name__ == "__main__":
    main()
