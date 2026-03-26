from pathlib import Path

ROOT_DIR = Path(__file__).resolve()
CKPT_DIR = ROOT_DIR.parent.parent.parent / "checkpoints"

FOLDER_SYNTHETIC = CKPT_DIR / "synthetic"
FOLDER_SYNTHETIC.mkdir(parents=True, exist_ok=True)
