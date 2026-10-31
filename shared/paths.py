"""Repository-local paths; speech audio lives outside the Git repository."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_root(value=None):
    return Path(value or os.environ.get('KWS_DATA_ROOT') or
                ROOT.parent / 'datasets' / 'speech_commands_v0.02').expanduser().resolve()


def ensure_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
