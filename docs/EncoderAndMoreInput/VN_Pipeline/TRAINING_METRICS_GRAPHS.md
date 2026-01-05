# Training Metrics & Graphs

This document explains the metrics tracked during seq2seq training and how to visualize them.

## Metrics Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         METRICS TRACKED DURING TRAINING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   During Training (every validation interval):                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   1. Training Loss                                              │       │
│   │      └── Cross-entropy loss on current batch                    │       │
│   │                                                                 │       │
│   │   2. Validation Loss                                            │       │
│   │      └── Cross-entropy loss on held-out validation set          │       │
│   │                                                                 │       │
│   │   3. Timestamp                                                  │       │
│   │      └── When this measurement was taken                        │       │
│   │                                                                 │       │
│   │   4. Step Number                                                │       │
│   │      └── Current training step                                  │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   After Training (via eval stub):                                           │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                 │       │
│   │   5. Length Ratio                                               │       │
│   │      └── len(output) / len(input)                               │       │
│   │      └── Ideal: ~1.0 (output same length as input)              │       │
│   │                                                                 │       │
│   │   6. Copy Match Rate                                            │       │
│   │      └── % of outputs that preserve non-highlighted text        │       │
│   │      └── Ideal: 1.0 (100% preservation)                         │       │
│   │                                                                 │       │
│   │   7. Exact Match Rate                                           │       │
│   │      └── % of outputs that exactly match targets                │       │
│   │      └── Ideal: 1.0 for copy task                               │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Where Metrics Are Saved

```
EncoderAndMoreInput/VN_Pipeline/out/seq2seq/
├── metrics.csv              ← Training metrics (loss vs step)
├── train.jsonl              ← Training data
├── val.jsonl                ← Validation data
├── best_checkpoint.pt       ← Best model (lowest val loss)
├── model_final.pt           ← Final model
└── plots/                   ← Generated graphs
    ├── val_loss.png
    ├── len_ratio.png
    ├── copy_match.png
    └── all_metrics.png
```

## CSV Format

```csv
step,timestamp,train_loss,val_loss,avg_len_ratio,copy_match_rate,exact_match_rate
0,2025-01-02 14:30:00,8.234,8.156,,,
100,2025-01-02 14:35:00,4.521,4.892,,,
200,2025-01-02 14:40:00,3.102,3.456,,,
...
```

## Graphs Explained

### 1. Validation Loss vs Step

```
   Val Loss
      │
   10 ┤ ●
      │  ╲
    8 ┤   ╲
      │    ╲
    6 ┤     ╲
      │      ╲
    4 ┤       ╲●───●
      │            ╲
    2 ┤             ╲●───●───●───●
      │
    0 ┼────────────────────────────────► Step
      0   1000  2000  3000  4000  5000

   GOOD: Steady decrease, then plateau
   BAD:  Flat line (not learning)
   BAD:  Goes up (overfitting)
```

### 2. Length Ratio vs Step

```
   Ratio
      │
   2.0 ┤
      │         (bad: output too long)
   1.5 ┤
      │
   1.1 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Acceptable zone
   1.0 ┤ ●───●───●───●───●───●───●─── IDEAL
   0.9 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Acceptable zone
      │
   0.5 ┤         (bad: output too short)
      │
   0.0 ┼────────────────────────────────► Step
      0   1000  2000  3000  4000  5000

   GOOD: Stays near 1.0
   BAD:  >> 1.0 (repetition/rambling)
   BAD:  << 1.0 (truncation/early stopping)
```

### 3. Copy Match Rate vs Step

```
   Rate
      │
   1.0 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ●───●───●─── PERFECT
      │               ╱
   0.8 ┤             ╱
      │           ╱
   0.6 ┤         ╱
      │       ╱
   0.4 ┤     ╱
      │   ╱
   0.2 ┤ ●
      │
   0.0 ┼────────────────────────────────► Step
      0   1000  2000  3000  4000  5000

   Measures: Does the model preserve text OUTSIDE <hl>...</hl>?
   
   GOOD: Approaches 1.0 (perfect preservation)
   BAD:  Stays low (model changes non-highlighted text)
```

### 4. Combined View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         all_metrics.png                                      │
├────────────────────────────────┬────────────────────────────────────────────┤
│                                │                                            │
│      Validation Loss           │         Length Ratio                       │
│                                │                                            │
│   ●                            │   ─ ─ ─ 1.1                                │
│    ╲                           │   ●───●───●───● 1.0                        │
│     ╲                          │   ─ ─ ─ 0.9                                │
│      ╲●───●                    │                                            │
│           ╲●                   │                                            │
│                                │                                            │
├────────────────────────────────┼────────────────────────────────────────────┤
│                                │                                            │
│      Copy Match Rate           │         Exact Match Rate                   │
│                                │                                            │
│   ─ ─ ─ 1.0 ●───●              │   ─ ─ ─ 1.0 ●───●                          │
│           ╱                    │           ╱                                │
│         ╱                      │         ╱                                  │
│       ╱                        │       ╱                                    │
│   ●──╱                         │   ●──╱                                     │
│                                │                                            │
└────────────────────────────────┴────────────────────────────────────────────┘
```

## How to Generate Graphs

### During Training (automatic)

The training loop now writes to `metrics.csv` automatically every validation interval.

### Generate Plots

```bash
# From repo root
python -m EncoderAndMoreInput.VN_Pipeline.eval.plot_seq2seq_metrics_stub
```

This creates PNG files in `out/seq2seq/plots/`.

### View Plots While Training

Open another terminal and run the plot script periodically:

```bash
# Watch plots update
while true; do
    python -m EncoderAndMoreInput.VN_Pipeline.eval.plot_seq2seq_metrics_stub
    sleep 60  # Update every minute
done
```

Or on Windows PowerShell:

```powershell
while ($true) {
    python -m EncoderAndMoreInput.VN_Pipeline.eval.plot_seq2seq_metrics_stub
    Start-Sleep -Seconds 60
}
```

## Interpreting Training Progress

### Good Training Signs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HEALTHY TRAINING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 0-1000:                                                              │
│   • Loss drops rapidly (10 → 3)                                             │
│   • Model learning basic structure                                          │
│                                                                             │
│   Step 1000-5000:                                                           │
│   • Loss continues dropping (3 → 1.5)                                       │
│   • Copy match rate increases                                               │
│   • Length ratio approaches 1.0                                             │
│                                                                             │
│   Step 5000+:                                                               │
│   • Loss plateaus around 0.5-1.5                                            │
│   • All metrics stable                                                      │
│   • Model has converged                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Warning Signs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PROBLEMS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ⚠️  Loss not decreasing:                                                  │
│       • Learning rate too low → increase it                                 │
│       • Architecture bug → check cross-attention                            │
│       • Data issue → verify train.jsonl format                              │
│                                                                             │
│   ⚠️  Val loss increasing while train loss decreases:                       │
│       • Overfitting → reduce learning rate, add regularization              │
│       • Train longer with early stopping                                    │
│                                                                             │
│   ⚠️  Loss stuck at ~3-4:                                                   │
│       • Cross-attention not learning → train from scratch                   │
│       • Model ignoring encoder → check architecture                         │
│                                                                             │
│   ⚠️  Length ratio >> 1:                                                    │
│       • Model generating repetitive text                                    │
│       • Add repetition penalty                                              │
│                                                                             │
│   ⚠️  Copy match rate stays at 0:                                           │
│       • Model not learning to preserve context                              │
│       • May need more training                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Recommended Training Settings

| Setting | Value | Notes |
|---------|-------|-------|
| `seq2seq_max_steps` | 10,000-20,000 | More is better for seq2seq |
| `seq2seq_learning_rate` | 1e-4 | For training from scratch |
| `seq2seq_val_interval` | 100 | Check validation every 100 steps |
| `seq2seq_save_interval` | 500 | Save checkpoint every 500 steps |
| `batch_size` | 8-16 | Higher if GPU memory allows |

## File Locations

| File | Purpose |
|------|---------|
| `training/seq2seq_train_stub.py` | Training loop (writes metrics.csv) |
| `eval/seq2seq_eval_stub.py` | Evaluation (computes all metrics) |
| `eval/plot_seq2seq_metrics_stub.py` | Generates PNG plots |
| `out/seq2seq/metrics.csv` | Raw metrics data |
| `out/seq2seq/plots/*.png` | Generated visualizations |

