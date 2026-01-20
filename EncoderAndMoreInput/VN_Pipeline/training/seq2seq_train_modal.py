"""
Modal training script for seq2seq model.
Clones repo from GitHub and trains on Modal GPU.
"""

import modal

app = modal.App("seq2seq-training")

volume = modal.Volume.from_name("chuni-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "tiktoken",
        "numpy",
    )
    .apt_install("git")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
)
def train():
    import subprocess
    import os
    import shutil
    from pathlib import Path

    repo_url = "https://github.com/sheepish-sheep/SmallLLm.git"
    repo_dir = Path("/root/repo")
    
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    
    print(f"Cloning {repo_url}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    
    os.chdir(repo_dir)
    
    data_dir = Path("/data")
    
    if (data_dir / "config.json").exists():
        shutil.copy(data_dir / "config.json", repo_dir / "config.json")
        print("Copied config.json from volume")
    
    seq2seq_out = repo_dir / "EncoderAndMoreInput/VN_Pipeline/out/seq2seq"
    seq2seq_out.mkdir(parents=True, exist_ok=True)
    
    if (data_dir / "seq2seq/train.jsonl").exists():
        shutil.copy(data_dir / "seq2seq/train.jsonl", seq2seq_out / "train.jsonl")
        shutil.copy(data_dir / "seq2seq/val.jsonl", seq2seq_out / "val.jsonl")
        print("Copied train/val data from volume")
    
    if (data_dir / "seq2seq/model_00250.pt").exists():
        shutil.copy(data_dir / "seq2seq/model_00250.pt", seq2seq_out / "model_00250.pt")
        print("Copied model_00250.pt from volume")
    
    log_dir = repo_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "model_04999_clean.pt").exists():
        shutil.copy(data_dir / "model_04999_clean.pt", log_dir / "model_04999_clean.pt")
        print("Copied base model from volume")
    
    finetune_dir = repo_dir / "EncoderAndMoreInput/VN_Pipeline/out/finetune_runs"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "finetune_runs/best_checkpoint.pt").exists():
        shutil.copy(data_dir / "finetune_runs/best_checkpoint.pt", finetune_dir / "best_checkpoint.pt")
        print("Copied VN finetune checkpoint from volume")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    import sys
    sys.path.insert(0, str(repo_dir))
    
    from EncoderAndMoreInput.VN_Pipeline.training import seq2seq_train_stub
    seq2seq_train_stub.main()
    
    print("\n" + "="*60)
    print("Saving results to volume...")
    print("="*60 + "\n")
    
    for pt_file in seq2seq_out.glob("*.pt"):
        shutil.copy(pt_file, data_dir / "seq2seq" / pt_file.name)
        print(f"Saved {pt_file.name}")
    
    if (seq2seq_out / "metrics.csv").exists():
        shutil.copy(seq2seq_out / "metrics.csv", data_dir / "seq2seq/metrics.csv")
        print("Saved metrics.csv")
    
    volume.commit()
    print("\nDone! Results saved to volume.")


@app.local_entrypoint()
def main():
    train.remote()
