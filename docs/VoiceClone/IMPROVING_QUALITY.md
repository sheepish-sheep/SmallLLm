# Improving Speech Quality

## Quick Controls (Overlay)
- Use the Expressiveness slider:
  - 0.0 = robotic/flat
  - 0.5 = balanced (default)
  - 1.0 = more animated/expressive
- Language dropdown: EN for English output, JA for Japanese output.

## Under the Hood (main.py)
- The slider maps to generation parameters:
  - `exaggeration`: 0.3 → 0.8 as expressiveness rises.
  - `cfg_weight`: 0.35 → 0.20 as expressiveness rises (lower = less robotic).
  - `temperature`: 0.75 → 1.10 as expressiveness rises (higher = more variation).
- `repetition_penalty` remains 2.0 to avoid loops.

## Suggested Ranges
- Natural and clear: expressiveness ~0.4–0.6.
- More dramatic: expressiveness ~0.7–0.9.
- Very steady: expressiveness ~0.2–0.4.

## Troubleshooting
- If audio is too monotone: increase expressiveness slightly and re-read.
- If audio sounds unstable: lower expressiveness and keep text shorter.
- If nothing plays: check overlay status messages; errors are shown in red.
