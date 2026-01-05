"""
Stub to auto-pick the newest model_*.pt checkpoint from log/.
"""

import re
from pathlib import Path
from typing import Optional


def find_latest_checkpoint(log_dir: str = "log") -> Optional[str]:
    """
    TODO: return the newest model_*.pt file path as a string.
    - Option A: choose by highest step number in filename.
    - Option B: choose by newest modified time.
    - Return None if no checkpoints found.
    """
    raise NotImplementedError("find_latest_checkpoint is a stub.")


def main() -> None:
    """
    TODO:
    - call find_latest_checkpoint()
    - print the path (so you can paste into config.json)
    """
    raise NotImplementedError("main is a stub.")


if __name__ == "__main__":
    main()
