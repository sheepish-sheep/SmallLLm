"""Analyze seq2seq training metrics and diagnose issues."""
import csv
from pathlib import Path
from collections import defaultdict

repo_root = Path(__file__).resolve().parents[3]
metrics_path = repo_root / "EncoderAndMoreInput/VN_Pipeline/out/seq2seq/metrics.csv"

# Load metrics
rows = []
with open(metrics_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            'step': int(row['step']),
            'train_loss': float(row['train_loss']),
            'val_loss': float(row['val_loss']),
        })

print(f"Total rows: {len(rows)}")
print(f"Step range: {rows[0]['step']} to {rows[-1]['step']}")

# Check for duplicate steps (indicates restarts)
step_counts = defaultdict(int)
for r in rows:
    step_counts[r['step']] += 1
duplicates = [s for s, c in step_counts.items() if c > 1]
if duplicates:
    print(f"\nWARNING: Found duplicate step entries (training restarts)")
    print(f"Duplicate steps: {sorted(duplicates)[:10]}...")  # Show first 10

# Keep only latest entry for each step
seen = {}
for r in rows:
    seen[r['step']] = r
rows = sorted(seen.values(), key=lambda x: x['step'])

print(f"\nAfter dedup: {len(rows)} rows")

# Key metrics analysis
print("\n" + "="*60)
print("TRAINING METRICS ANALYSIS")
print("="*60)

# Initial loss (should be ~10-11 for random init, ~4-5 for pretrained)
initial_loss = rows[0]['train_loss']
print(f"\nInitial train loss: {initial_loss:.2f}")
if initial_loss > 8:
    print("  -> HIGH initial loss suggests model started from SCRATCH (not pretrained)")
elif initial_loss < 5:
    print("  -> LOW initial loss suggests model was initialized from pretrained weights")

# Check overfitting gap
final_rows = rows[-20:]
avg_train = sum(r['train_loss'] for r in final_rows) / len(final_rows)
avg_val = sum(r['val_loss'] for r in final_rows) / len(final_rows)
gap = avg_val - avg_train
print(f"\nFinal 20 steps average:")
print(f"  Train loss: {avg_train:.3f}")
print(f"  Val loss: {avg_val:.3f}")
print(f"  Gap (val - train): {gap:.3f}")

if gap > 1.5:
    print("  -> SEVERE overfitting! Model memorized training data")
elif gap > 0.8:
    print("  -> MODERATE overfitting")
else:
    print("  -> Overfitting within acceptable range")

# Best validation loss
best_row = min(rows, key=lambda x: x['val_loss'])
print(f"\nBest validation loss: {best_row['val_loss']:.3f} at step {best_row['step']}")

# Check if val loss plateaued or got worse
mid_point = len(rows) // 2
first_half_val = sum(r['val_loss'] for r in rows[:mid_point]) / mid_point
second_half_val = sum(r['val_loss'] for r in rows[mid_point:]) / (len(rows) - mid_point)
print(f"\nVal loss trend:")
print(f"  First half avg: {first_half_val:.3f}")
print(f"  Second half avg: {second_half_val:.3f}")
if second_half_val < first_half_val:
    print("  -> Val loss improved in second half (good)")
else:
    print("  -> Val loss got WORSE in second half (potential collapse)")

# Show loss progression at key points
print(f"\nLoss progression:")
checkpoints = [0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1]
for idx in checkpoints:
    r = rows[idx]
    print(f"  Step {r['step']:5d}: train={r['train_loss']:.3f}, val={r['val_loss']:.3f}, gap={r['val_loss']-r['train_loss']:.3f}")

# Summary diagnosis
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)
issues = []

if initial_loss > 8:
    issues.append("Model trained from SCRATCH (not from pretrained weights)")
if gap > 1.0:
    issues.append(f"Severe overfitting (gap={gap:.2f})")
if avg_train < 1.5:
    issues.append(f"Train loss very low ({avg_train:.2f}) - model may have memorized data")
if best_row['val_loss'] > 2.3:
    issues.append(f"Best val loss still high ({best_row['val_loss']:.2f}) - model didn't learn task well")

if issues:
    print("\nPROBLEMS FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n" + "="*60)
    print("ROOT CAUSE ANALYSIS")
    print("="*60)

    if initial_loss > 8:
        print("""
The model started with ~9.1 initial loss, which is characteristic of
RANDOM INITIALIZATION, not transfer learning from pretrained weights.

For comparison:
- Random init: loss starts ~10-11 (uniform over 50k vocab)
- Pretrained GPT: loss starts ~4-5

This means the VN-finetuned weights were NOT successfully loaded.
The model had to learn language modeling FROM SCRATCH while also
learning the paraphrasing task - which is extremely difficult.
""")

    if gap > 1.0:
        print(f"""
The train-val gap of {gap:.2f} indicates severe overfitting.
Train loss: {avg_train:.2f} (memorized training examples)
Val loss: {avg_val:.2f} (cannot generalize)

This causes REPETITIVE OUTPUT because:
1. Model memorized specific token sequences from training
2. When given new input, it falls back to common patterns it memorized
3. These patterns loop because they were frequent in training data
""")

else:
    print("\nNo major issues detected in metrics.")

print("\n" + "="*60)
