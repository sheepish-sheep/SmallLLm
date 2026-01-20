"""
Filtered Seq2Seq Rewriter - Generates multiple candidates and picks the best one.

This combines the seq2seq model with output filtering to improve quality.

Usage:
    from EncoderAndMoreInput.VN_Pipeline.inference.filtered_rewriter import FilteredRewriter

    rewriter = FilteredRewriter()
    result = rewriter.rewrite("I love you so much.")

    print(result.output)       # Best output
    print(result.passed)       # Whether it passed filters
    print(result.attempts)     # Number of candidates generated
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
from dataclasses import dataclass
from typing import Any

import torch

from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import (
    load_seq2seq_model,
    load_checkpoint,
    generate_replacement,
)
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding
from EncoderAndMoreInput.VN_Pipeline.utils.output_filter import OutputFilter, FilterResult


@dataclass
class RewriteResult:
    """Result from filtered rewriting."""
    output: str
    input_text: str
    passed: bool
    attempts: int
    all_candidates: list[str]
    rejection_reasons: list[str]


class FilteredRewriter:
    """
    Seq2seq rewriter with output filtering.

    Generates multiple candidates and picks the best one that passes filters.
    """

    def __init__(
        self,
        config_path: Path = None,
        checkpoint_path: Path = None,
        num_candidates: int = 3,
        temperatures: list[float] = None,
        filter_config: dict = None,
    ):
        """
        Initialize the filtered rewriter.

        Args:
            config_path: Path to config.json (default: repo_root/config.json)
            checkpoint_path: Path to model checkpoint (default: from config)
            num_candidates: Number of candidates to generate per input
            temperatures: List of temperatures to try (default: [0.6, 0.7, 0.8])
            filter_config: Custom filter settings (passed to OutputFilter)
        """
        if config_path is None:
            config_path = repo_root / "config.json"

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.num_candidates = num_candidates
        self.temperatures = temperatures or [0.6, 0.7, 0.8]

        # Load model
        self.model = load_seq2seq_model(self.config, repo_root)

        if checkpoint_path is None:
            checkpoint_path = self.config.get("seq2seq_checkpoint_path")
            if checkpoint_path:
                checkpoint_path = repo_root / checkpoint_path
            else:
                checkpoint_path = repo_root / self.config["seq2seq_output_dir"] / "best_checkpoint.pt"

        load_checkpoint(self.model, checkpoint_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Tokenizer
        self.enc = build_hl_encoding()
        self.source_prefix = self.config.get("seq2seq_source_prefix", "")
        self.source_suffix = self.config.get("seq2seq_source_suffix", "")
        self.repetition_penalty = float(self.config.get("seq2seq_repetition_penalty", 1.2))

        # Filter
        filter_config = filter_config or {}
        self.filter = OutputFilter(**filter_config)

    def _generate_one(self, text: str, temperature: float, max_len: int) -> str:
        """Generate a single candidate."""
        return generate_replacement(
            text=text,
            model=self.model,
            enc=self.enc,
            max_len=max_len,
            temperature=temperature,
            top_k=40,
            device=self.device,
            source_prefix=self.source_prefix,
            source_suffix=self.source_suffix,
            repetition_penalty=self.repetition_penalty,
        )

    def rewrite(
        self,
        text: str,
        max_len: int = None,
        num_candidates: int = None,
    ) -> RewriteResult:
        """
        Rewrite text with filtering.

        Args:
            text: Input text to paraphrase
            max_len: Maximum output length (default: based on input length)
            num_candidates: Override number of candidates to generate

        Returns:
            RewriteResult with best output and metadata
        """
        if max_len is None:
            # Calculate based on input length
            len_ratio = float(self.config.get("seq2seq_len_ratio", 1.3))
            min_gen_len = int(self.config.get("seq2seq_min_gen_len", 8))
            max_gen_len = int(self.config.get("seq2seq_max_gen_len", 80))
            source_tokens = self.enc.encode(f"{self.source_prefix}{text}{self.source_suffix}")
            max_len = min(max_gen_len, max(min_gen_len, int(len(source_tokens) * len_ratio)))

        num_candidates = num_candidates or self.num_candidates
        candidates = []

        # Generate candidates with different temperatures
        for i in range(num_candidates):
            temp = self.temperatures[i % len(self.temperatures)]
            try:
                candidate = self._generate_one(text, temp, max_len)
                candidates.append(candidate)
            except Exception as e:
                print(f"Generation failed: {e}")
                continue

        if not candidates:
            return RewriteResult(
                output=text,  # Fallback to input
                input_text=text,
                passed=False,
                attempts=0,
                all_candidates=[],
                rejection_reasons=["All generation attempts failed"],
            )

        # Filter and pick best
        result = self.filter.filter_with_fallback(text, candidates)

        return RewriteResult(
            output=result.output,
            input_text=text,
            passed=result.passed,
            attempts=len(candidates),
            all_candidates=candidates,
            rejection_reasons=result.rejection_reasons,
        )

    def rewrite_batch(self, texts: list[str], **kwargs) -> list[RewriteResult]:
        """Rewrite multiple texts."""
        return [self.rewrite(t, **kwargs) for t in texts]


# =============================================================================
# MAIN - Demo and test
# =============================================================================

if __name__ == "__main__":
    print("Loading FilteredRewriter...")
    rewriter = FilteredRewriter(num_candidates=3)
    print(f"Device: {rewriter.device}")
    print()

    test_inputs = [
        "I love you so much.",
        "What are you doing here?",
        "She walked into the room slowly.",
        "The moon was shining brightly tonight.",
        "Why did you leave without saying goodbye?",
        "I am so happy right now!",
        "He picked up the sword.",
        "Thank you for everything.",
    ]

    print("=" * 70)
    print("FILTERED REWRITER TEST")
    print("=" * 70)

    for text in test_inputs:
        result = rewriter.rewrite(text)

        print(f"\nInput:  {text}")
        print(f"Output: {result.output}")
        print(f"Passed: {result.passed} | Attempts: {result.attempts}")

        if not result.passed:
            print(f"Issues: {result.rejection_reasons}")

        if len(result.all_candidates) > 1:
            print("All candidates:")
            for i, c in enumerate(result.all_candidates):
                marker = " <-- selected" if c == result.output else ""
                print(f"  [{i+1}] {c}{marker}")

        print("-" * 70)
