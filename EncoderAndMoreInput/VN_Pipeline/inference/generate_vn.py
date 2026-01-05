import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import tiktoken
import __main__


def load_train_chunni(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    train_chunni_path = repo_root / "train-chunni.py"
    spec = importlib.util.spec_from_file_location("train_chunni", train_chunni_path)
    train_chunni = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_chunni)
    return train_chunni


def load_base_checkpoint(path: Path, train_chunni):
    __main__.GPTConfig = train_chunni.GPTConfig
    if hasattr(torch.serialization, "safe_globals"):
        original_module = train_chunni.GPTConfig.__module__
        train_chunni.GPTConfig.__module__ = "__main__"
        try:
            with torch.serialization.safe_globals([train_chunni.GPTConfig]):
                return torch.load(path, map_location="cpu", weights_only=False)
        finally:
            train_chunni.GPTConfig.__module__ = original_module
    return torch.load(path, map_location="cpu")


def load_finetune_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def sample(model, enc, prompt, steps, temperature, top_k, device):
    tokens = enc.encode(prompt)
    if len(tokens) > model.config.block_size:
        tokens = tokens[-model.config.block_size:]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x_cond = x[:, -model.config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_id = indices.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)
    return enc.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with the VN fine-tuned model.")
    parser.add_argument("--prompt", default="", help="Prompt text. If empty, read from stdin.")
    parser.add_argument("--steps", type=int, default=80, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    config = json.loads((repo_root / "config.json").read_text(encoding="utf-8"))

    base_ckpt_path = repo_root / config["base_checkpoint_path"]
    if not base_ckpt_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")

    resume_path = config.get("finetune_resume_path") or "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs/best_checkpoint.pt"
    resume_path = repo_root / resume_path
    if not resume_path.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {resume_path}")

    train_chunni = load_train_chunni(repo_root)
    base_ckpt = load_base_checkpoint(base_ckpt_path, train_chunni)
    model = train_chunni.GPT(base_ckpt["config"])

    ft_ckpt = load_finetune_checkpoint(resume_path)
    model.load_state_dict(ft_ckpt["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    enc = tiktoken.get_encoding("gpt2")

    prompt = args.prompt
    if not prompt:
        prompt = input("Prompt: ").strip()
    output = sample(model, enc, prompt, args.steps, args.temperature, args.top_k, device)
    print(output)


if __name__ == "__main__":
    main()
