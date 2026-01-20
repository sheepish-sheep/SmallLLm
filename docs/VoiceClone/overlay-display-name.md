# Overlay Display Name Override

You can change the overlay’s display name (e.g., show “Marie” locally) without committing it to the repo by using the `VOICE_DISPLAY_NAME` environment variable.

## Windows (PowerShell)

```powershell
$env:VOICE_DISPLAY_NAME = "Marie"
python main.py
```

## macOS/Linux (bash/zsh)

```bash
VOICE_DISPLAY_NAME="Marie" python main.py
```

Notes:
- The default name remains `Sample Voice` when no env var is set.
- `.env` is ignored by Git, so you can also set `VOICE_DISPLAY_NAME=Marie` in a local `.env` if you load it yourself before starting the app.
