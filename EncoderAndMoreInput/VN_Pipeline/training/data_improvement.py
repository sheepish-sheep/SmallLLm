import json
import sys
from pathlib import Path
from dataclasses import dataclass
import torch

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@dataclass
class ImprovementConfig:
    input_path: str = "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/train.jsonl"
    output_path: str = "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/improved_train.jsonl"
    num_candidates: int = 5
    temperatures: list = None
    max_gen_length: int = 50
    min_output_length: int = 5
    max_length_ratio: float = 2.5
    require_all_filters: bool = True
    batch_size: int = 32
    max_samples: int = 0
    skip_existing: bool = True


def load_training_data(input_path: str) -> list[dict]:
    pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            pairs.append({"source": data["source"]})
    return pairs


def load_model():
    from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import (
        load_seq2seq_model, load_checkpoint
    )
    from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding
    config = json.loads((repo_root / "config.json").read_text())
    model = load_seq2seq_model(config, repo_root)
    checkpoint_path = config.get("seq2seq_checkpoint_path", "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/best_checkpoint.pt")
    load_checkpoint(model, repo_root / checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = build_hl_encoding()
    return model, tokenizer, device


def generate_candidates(
    model,
    tokenizer,
    device: str,
    input_text: str,
    num_candidates: int,
    temperatures: list[float],
    max_length: int,
) -> list[str]:
    from EncoderAndMoreInput.VN_Pipeline.inference.seq2seq_infer import generate_replacement
    candidates = []
    for i in range(num_candidates):
        temp = temperatures[i % len(temperatures)]
        output = generate_replacement(
            text=input_text,
            model=model,
            enc=tokenizer,
            max_len=max_length,
            temperature=temp,
            top_k=40,
            device=device,
            source_prefix="",
            source_suffix="",
            repetition_penalty=1.2,
        )
        candidates.append(output)
    return candidates


def filter_candidates(
    input_text: str,
    candidates: list[str],
    config: ImprovementConfig,
) -> list[str]:
    from EncoderAndMoreInput.VN_Pipeline.utils.output_filter import OutputFilter
    filter = OutputFilter(
        min_length=config.min_output_length,
        max_length_ratio=config.max_length_ratio,
    )
    passing = []
    for candidate in candidates:
        result = filter.filter(input_text, candidate)
        if result.passed:
            passing.append(candidate)
    return passing


def select_best_candidate(
    input_text: str,
    passing_candidates: list[str],
) -> str | None:
    if not passing_candidates:
        return None
        #should change most likely based on the criteria(Will i remember this?)
    return min(passing_candidates, key=len)


def save_improved_pair(
    output_file,
    source: str,
    target: str,
) -> None:
    output_file.write(json.dumps({"source": source, "target": target}, ensure_ascii=False) + "\n")


def run_improvement_pipeline(config: ImprovementConfig) -> dict:
    stats = {"total": 0, "success": 0, "failed": 0}
    model, tokenizer, device = load_model()
    data = load_training_data(config.input_path)
    temperatures = config.temperatures or [0.6, 0.7, 0.8]
    with open(config.output_path, 'w', encoding='utf-8') as out_f:
        for item in data:
            input_text = item["source"]
            candidates = generate_candidates(model, tokenizer, device, input_text, config.num_candidates, temperatures, config.max_gen_length)
            passing = filter_candidates(input_text, candidates, config)
            best = select_best_candidate(input_text, passing)
            if best:
                save_improved_pair(out_f, input_text, best)
                stats["success"] += 1
            else:
                stats["failed"] += 1
            stats["total"] += 1
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--num-candidates", type=int, default=5)
    args = parser.parse_args()
    config = ImprovementConfig(
        input_path=args.input,
        output_path=args.output,
        num_candidates=args.num_candidates,
    )
    stats = run_improvement_pipeline(config)
    print(f"Done! Success rate: {stats['success']/stats['total']*100:.1f}%")


if __name__ == "__main__":
    main()
