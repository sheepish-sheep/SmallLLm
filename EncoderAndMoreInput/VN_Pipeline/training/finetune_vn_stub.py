"""
Stub for fine-tuning your base model on VN dialogue.

Use `train-chunni.py` and the TrainingLoop docs as reference,
but do not paste the full training loop here until you are ready.
"""

import importlib.util
import json
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Optional
import __main__

import tiktoken
import torch

def load_train_chunni_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    train_chunni_path = repo_root / "train-chunni.py"
    spec = importlib.util.spec_from_file_location("train_chunni", train_chunni_path)
    train_chunni = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_chunni)
    return train_chunni

def load_base_checkpoint(checkpoint_path: str) -> Any:
    """
    Load a checkpoint trained on FineWeb (or your latest checkpoint).
    Return a (model, start_step) tuple you will fine-tune.
    - Use torch.load(checkpoint_path, map_location="cpu") first.
    - Rebuild the model from the saved config, then load state_dict.
    """
    log_dir = "log"
    if not os.path.isdir(log_dir):
        raise FileNotFoundError("Log directory not found")
    checkpoint_files = [f for f in os.listdir(log_dir)
                    if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoint_files and not checkpoint_path:
        raise FileNotFoundError("No checkpoints found")
    if not checkpoint_path:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        latest = checkpoint_files[-1]
        checkpoint_path = os.path.join(log_dir, latest)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found")
    train_chunni = load_train_chunni_module()
    __main__.GPTConfig = train_chunni.GPTConfig
    __main__.GPT = train_chunni.GPT
    if hasattr(torch.serialization, "safe_globals"):
        original_module = train_chunni.GPTConfig.__module__
        train_chunni.GPTConfig.__module__ = "__main__"
        try:
            with torch.serialization.safe_globals([train_chunni.GPTConfig]):
                try:
                    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                except Exception:
                    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        finally:
            train_chunni.GPTConfig.__module__ = original_module
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    if cfg is None:
        raise ValueError("Config not found in checkpoint")
    model = train_chunni.GPT(cfg)
    model.load_state_dict(ckpt["model"])
    start_step = ckpt.get("step", 0)
    return model, start_step

def build_vn_dataloader(vn_text_path: str, batch_size: int, seq_len: int) -> Any:
    """
    Tokenize VN text and build batches.
    """
    data_root = Path(vn_text_path)
    if not data_root.is_dir():
        raise FileNotFoundError("VN shard directory not found")

    train_chunni = load_train_chunni_module()

    train_loader = train_chunni.DataLoader(
        B=batch_size,
        T=seq_len,
        process_rank=0,
        num_processes=1,
        split="train",
        data_root=str(data_root),
    )
    val_loader = train_chunni.DataLoader(
        B=batch_size,
        T=seq_len,
        process_rank=0,
        num_processes=1,
        split="val",
        data_root=str(data_root),
    )
    return train_loader, val_loader


def train_finetune_loop(
    model: Any,
    dataloader: Any,
    output_dir: str,
    max_steps: int = 3000,
    learning_rate: float = 5e-5,
    val_steps: int = 10,
    start_step: int = 0,
    resume_state: Optional[dict] = None,
) -> None:
    """
    TODO: Implement a fine-tuning loop:
    - low learning rate (1e-5 to 5e-5)
    - shorter max steps / epochs
    - regular validation sampling
    - save best checkpoint to output_dir
    """
    if dataloader is None:
        raise ValueError("Dataloader is required")
    if output_dir is None:
        raise ValueError("Output directory is required")
    if model is None:
        raise ValueError("Model is required")
    if isinstance(dataloader, tuple):
        train_loader, val_loader = dataloader
    else:
        train_loader = dataloader
        val_loader = None
    max_lr = learning_rate
    min_lr = max_lr * 0.1
    grad_accum_steps = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    if resume_state:
        optimizer_state = resume_state.get("optimizer")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=min_lr,
        last_epoch=start_step - 1,
    )
    if resume_state:
        scheduler_state = resume_state.get("scheduler")
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    best_val_loss = float('inf')
    val_loss = None
    os.makedirs(output_dir, exist_ok=True)
    for step in range(start_step, max_steps):
        loss_accum = torch.tensor(0.0, device=device)
        model.train()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        del logits, loss
        if device_type == "cuda":
            torch.cuda.empty_cache()

        if val_loader is not None and step % 500 == 0:
            val_loss = evaluate(model, val_loader, device, steps=val_steps)
            print(f"Step {step} | Val Loss: {val_loss:.6f}")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                'model': model.state_dict(),
                'step': step,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            output_path = os.path.join(output_dir, "best_checkpoint.pt")
            torch.save(best_checkpoint, output_path)
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss_accum.item():.6f}")
            checkpoint_path = os.path.join(output_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'step': step,
                'val_loss': None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

def evaluate(model, val_loader, device, steps=3):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total += loss.item()
    model.train()
    return total / steps


def main() -> None:
    """
    - Load config.json from repo root.
    - Load checkpoint.
    - Build VN dataloader.
    - Run fine-tune loop.
    - Save best checkpoint to output dir.
    """
    config_path = Path(__file__).resolve().parents[3] / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Config file not found")
    with open(config_path, "r") as f:
        config = json.load(f)
    model, start_step = load_base_checkpoint(config["base_checkpoint_path"])
    start_step = 0
    resume_path = config.get("finetune_resume_path")
    resume_state = None
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = Path(__file__).resolve().parents[3] / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"finetune_resume_path not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_ckpt["model"])
        start_step = resume_ckpt.get("step", start_step) + 1
        resume_state = {
            "optimizer": resume_ckpt.get("optimizer"),
            "scheduler": resume_ckpt.get("scheduler"),
        }
    output_dir = config.get("finetune_output_dir")
    if not output_dir:
        raise ValueError("finetune_output_dir is required in config.json")
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = Path(__file__).resolve().parents[3] / output_dir_path
    if output_dir_path.exists() and any(output_dir_path.iterdir()):
        run_suffix = time.strftime("run_%Y%m%d_%H%M%S")
        output_dir_path = output_dir_path / run_suffix
        print(f"Output directory not empty, using {output_dir_path}")
    output_dir = str(output_dir_path)
    shard_dir = config.get("vn_shard_dir")
    if not shard_dir:
        raise ValueError("vn_shard_dir is required in config.json (directory with train_*.npy and val_*.npy)")
    seq_len = config["max_seq_len"]
    if seq_len > model.config.block_size:
        print(f"max_seq_len {seq_len} > model block_size {model.config.block_size}; clamping to block_size.")
        seq_len = model.config.block_size
    train_loader, val_loader = build_vn_dataloader(shard_dir, config["batch_size"], seq_len)
    max_steps = config.get("finetune_max_steps", 3000)
    learning_rate = float(config.get("learning_rate", 5e-5))
    lr_min = 1e-5
    lr_max = 5e-5
    if learning_rate < lr_min or learning_rate > lr_max:
        print(f"learning_rate {learning_rate:.2e} is outside [{lr_min:.0e}, {lr_max:.0e}], clamping.")
        learning_rate = min(max(learning_rate, lr_min), lr_max)
    val_steps = config.get("finetune_val_steps", 10)
    train_finetune_loop(
        model,
        (train_loader, val_loader),
        output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        val_steps=val_steps,
        start_step=start_step,
        resume_state=resume_state,
    )
    if val_loader is not None:
        best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
        if os.path.isfile(best_checkpoint_path):
            best_alias_path = Path(output_dir).parent / "best_checkpoint.pt"
            shutil.copy2(best_checkpoint_path, best_alias_path)
            print(f"Best validation checkpoint copied to {best_alias_path}")
        best_checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(best_checkpoint["model"])
        print(f"Best validation loss: {best_checkpoint['val_loss']:.6f}")
        print(f"Best validation step: {best_checkpoint['step']}")
        print(f"Best validation checkpoint saved to {os.path.join(output_dir, 'best_checkpoint.pt')}")
    else:
        print("No validation data provided")


if __name__ == "__main__":
    main()
