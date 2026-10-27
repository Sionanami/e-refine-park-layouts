"""Repository roots for the code_1024 package."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = CODE_ROOT.parent
LEGACY_ROOT = CODE_ROOT / 'original_reference'
PARAMETERS_PATH = Path(__file__).resolve().parent / 'params' / 'parameters.json'


def _summary_dir(prefix: str) -> Path:
    matches = [p for p in EXPERIMENT_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {prefix}* directory under {EXPERIMENT_ROOT}, found {matches}')
    return matches[0]


def r1_summary_dir() -> Path:
    return _summary_dir('R1_')


def r2_summary_dir() -> Path:
    return _summary_dir('R2_')


def ensure_code_root_on_path() -> Path:
    root = str(CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return CODE_ROOT
