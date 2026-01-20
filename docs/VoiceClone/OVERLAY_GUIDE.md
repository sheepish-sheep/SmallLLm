# Transparent Overlay Window Guide

## Keybinds
- Read highlighted text: `Ctrl+Alt+Shift+V` (auto-copies selection)
- Show/Hide overlay: `Ctrl+Alt+Shift+O`

## How to Use
1. Highlight text anywhere (no need to press Ctrl+C).
2. Press `Ctrl+Alt+Shift+V` to read immediately.
3. Press `Ctrl+Alt+Shift+O` to show the overlay if you want on-screen controls.
4. In the overlay you can:
   - Toggle **Language** between English (en) and Japanese (ja).
   - Adjust the **Expressiveness** slider (0 = robotic, 1 = expressive).
   - Click **Hide Overlay** to dismiss it.

## Overlay Behavior
- Semi-transparent, always-on-top, borderless window.
- Runs its own UI thread so it appears reliably as soon as created.
- Status label shows Ready, Generating, Playing, or Error with colors.

## Customization
- Change keybinds in `config.py` (`READ_KEYBIND`, `OVERLAY_KEYBIND`).
- Adjust transparency or position in `overlay_window.py` (`attributes('-alpha', ...)`, `geometry(...)`).
- Add characters in `overlay_window.py` under `self.characters`.

## Troubleshooting
- Overlay not visible: press `Ctrl+Alt+Shift+O` again; it starts hidden but shows once created.
- Nothing reads: ensure text is highlighted; the app auto-copies but empty selections return "No text found".
- GPU not used: confirm PyTorch CUDA build is installed (see `INSTALL_GPU_PYTORCH.md`).
