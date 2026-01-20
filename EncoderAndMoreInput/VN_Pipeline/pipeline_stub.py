"""
Pipeline stages:
1. Data prep: Create copy/replace pairs from VN dialogue
2. Fine-tune: Fine-tune base model on VN text
3. Seq2seq train: Train encoder-decoder for paraphrasing
4. Seq2seq eval: Evaluate model quality
5. Voice: Synthesize audio from text
6. Rewrite+Voice: Full pipeline - paraphrase text then synthesize speech
"""

import json
from pathlib import Path
import importlib
import sys


def main(stages: list[str] = None) -> None:
    """
    Run the VN pipeline.
    
    Args:
        stages: Optional list of stages to run. If None, runs all stages.
                Valid stages: "data_prep", "finetune", "seq2seq", "eval", "voice"
    """
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Config file not found")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Validate config
    from EncoderAndMoreInput.VN_Pipeline.utils.config_schema import validate_config
    errors = validate_config(config, required_only=True)
    if errors:
        print("Config validation errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nFix these errors in config.json before running the pipeline.")
        sys.exit(1)
    
    # Import modules (now in subfolders)
    data_prep = importlib.import_module("EncoderAndMoreInput.VN_Pipeline.training.data_prep_stub")
    finetune_vn = importlib.import_module("EncoderAndMoreInput.VN_Pipeline.training.finetune_vn_stub")
    seq2seq_train = importlib.import_module("EncoderAndMoreInput.VN_Pipeline.training.seq2seq_train_stub")
    seq2seq_eval = importlib.import_module("EncoderAndMoreInput.VN_Pipeline.eval.seq2seq_eval_stub")
    voice = importlib.import_module("EncoderAndMoreInput.VN_Pipeline.voice_stub")

    # Training stages (run by default with "all")
    training_stages = ["data_prep", "finetune", "seq2seq", "eval", "voice"]
    # Inference stage (run explicitly)
    inference_stages = ["rewrite_voice"]
    if stages is None:
        stages = training_stages
    
    if "data_prep" in stages:
        print("\n=== Stage 1: Data Preparation ===")
        data_prep.main()
    
    if "finetune" in stages:
        print("\n=== Stage 2: Fine-tuning ===")
        finetune_vn.main()
    
    if "seq2seq" in stages:
        print("\n=== Stage 3: Seq2Seq Training ===")
        seq2seq_train.main()
    
    if "eval" in stages:
        print("\n=== Stage 4: Evaluation ===")
        seq2seq_eval.main()
    
    if "voice" in stages:
        print("\n=== Stage 5: Voice Synthesis ===")
        voice_texts = config.get("pipeline_voice_texts")
        if voice_texts:
            for text in voice_texts:
                voice.speak_text(text, config)
        else:
            voice.main()

    if "rewrite_voice" in stages:
        print("\n=== Stage 6: Rewrite + Voice (Full Pipeline) ===")
        from EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak import (
            rewrite_and_speak,
            batch_rewrite_and_speak,
        )
        voice_texts = config.get("pipeline_voice_texts")
        if voice_texts:
            results = batch_rewrite_and_speak(voice_texts)
            print(f"\nProcessed {len(results)} texts")
            for r in results:
                print(f"  {r['input'][:40]}... -> {r['audio_path']}")
        else:
            # Interactive mode
            print("No pipeline_voice_texts in config. Use interactive mode:")
            print("  python -m EncoderAndMoreInput.VN_Pipeline.inference.rewrite_and_speak -i")

    print("\n=== Pipeline complete ===")


def cli() -> None:
    """CLI entry point with stage selection."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the VN Pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["data_prep", "finetune", "seq2seq", "eval", "voice", "rewrite_voice", "all"],
        default=["all"],
        help="Stages to run (default: all)"
    )
    args = parser.parse_args()

    # "all" doesn't include rewrite_voice by default (it's an inference stage)
    if "all" in args.stages:
        stages = ["data_prep", "finetune", "seq2seq", "eval", "voice"]
    else:
        stages = args.stages
    main(stages)


if __name__ == "__main__":
    cli()
