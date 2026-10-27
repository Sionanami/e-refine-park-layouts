# -*- coding: utf-8 -*-
"""Shared constants and helpers for overnight_fixed_v2."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

EXP = Path(os.environ.get("SD15_EXP", "."))
RUN_ROOT = EXP / "runs"
MODEL = Path(os.environ.get("SD15_MODEL", "stable-diffusion-v1-5"))
FIXED_MAN = Path(os.environ.get("SD15_MANIFEST", "manifests"))
DATA_FIXED = Path(os.environ.get("SD15_DATA", "data"))
PIX2PIX_VAL = Path(os.environ.get("SD15_PIX2PIX_VAL", "pix2pix_val"))
PY = Path(os.environ.get("PYTHON", "python"))

PROMPT = (
    "top-down outdoor park layout map, flat discrete color blocks, "
    "semantic plan with green space, plant, pavement, building, water, and park roads, "
    "no photo, no perspective rendering"
)
NEGATIVE_PROMPT = ""
SEEDS_PANEL = [42, 2026]
INFER_STEPS = 20
CFG = 7.5
RESOLUTION = 512

# C4 = rotations only (legacy). C8 = D4 dihedral (rotations + horizontal-mirror∘rot).
# Spec TRAINING_SPEC.md requires ≥500 train pairs → default C8 (89×8=712).
TRANSFORM_IDS_C4 = ["r0", "r90", "r180", "r270"]
TRANSFORM_IDS_C8 = TRANSFORM_IDS_C4 + ["m_r0", "m_r90", "m_r180", "m_r270"]
TRANSFORM_IDS = list(TRANSFORM_IDS_C8)  # default for ≥500 offline pairs
MIRROR_ALLOWED = True
MIRROR_BLOCK_REASON = (
    "Legacy note: older runs used C4 only (no_flip) for orientation sensitivity. "
    "Current TRAINING_SPEC requires at least 500 augmented train pairs; C8 (D4) enabled. "
    "Independent site count remains 89; val/test stay unaugmented and disjoint."
)
TARGET_MIN_TRAIN_PAIRS = 500


def ensure_run_dirs(root: Path) -> dict[str, Path]:
    d = {
        "root": root,
        "code": root / "code",
        "data_aug": root / "data_aug",
        "manifests": root / "manifests",
        "preflight": root / "preflight",
        "stage1_lora": root / "stage1_lora",
        "stage2_adapter": root / "stage2_adapter",
        "eval": root / "eval",
        "logs": root / "logs",
        "fixed_eval_noise": root / "fixed_eval_noise",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_bytes_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_geom(img, transform_id: str):
    """Lossless discrete transforms via PIL transpose (no interpolation)."""
    from PIL import Image

    t = transform_id
    if t == "r0":
        return img
    if t == "r90":
        return img.transpose(Image.ROTATE_90)
    if t == "r180":
        return img.transpose(Image.ROTATE_180)
    if t == "r270":
        return img.transpose(Image.ROTATE_270)
    if t.startswith("m_"):
        # mirror then rotate - disabled by policy; kept for completeness
        base = img.transpose(Image.FLIP_LEFT_RIGHT)
        return apply_geom(base, t[2:])
    raise ValueError(f"unknown transform_id={transform_id}")


def pil_to_b_tensor(img, resolution: int) -> torch.Tensor:
    from PIL import Image

    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.NEAREST)
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)) * 2 - 1


def pil_to_a_tensor(img, resolution: int) -> torch.Tensor:
    from PIL import Image

    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.NEAREST)
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def optimizer_lr_snapshot(opt) -> dict:
    return {
        "optimizer_type": type(opt).__name__,
        "optimizer_module": type(opt).__module__,
        "param_groups": [
            {
                "lr": float(g.get("lr", 0.0)),
                "betas": list(g["betas"]) if "betas" in g else None,
                "eps": float(g["eps"]) if "eps" in g else None,
                "weight_decay": float(g.get("weight_decay", 0.0)),
                "momentum": float(g["momentum"]) if "momentum" in g else None,
                "n_params": sum(p.numel() for p in g["params"]),
            }
            for g in opt.param_groups
        ],
    }


def trainable_param_vector(module, name_substr: str | None = None) -> torch.Tensor:
    chunks = []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if name_substr is not None and name_substr.lower() not in n.lower():
            continue
        chunks.append(p.detach().float().reshape(-1).cpu())
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)


def vector_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).sum().sqrt().item())


def param_norm(module, name_substr: str | None = None) -> float:
    v = trainable_param_vector(module, name_substr)
    return float(v.pow(2).sum().sqrt().item() / math.sqrt(max(v.numel(), 1)))


def diffusion_target(noise_scheduler, noise, latents, timesteps):
    """Match scheduler prediction_type."""
    pt = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if pt == "epsilon":
        return noise, pt
    if pt == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps), pt
    if pt == "sample":
        return latents, pt
    raise ValueError(f"unsupported prediction_type={pt}")


def save_rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        st["cuda"] = [x.tolist() for x in torch.cuda.get_rng_state_all()]
    # python getstate has nested structures; json-safe via pickle file preferred
    return st


def dump_rng(path: Path) -> None:
    import pickle

    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(st, f)


def load_rng(path: Path) -> None:
    import pickle

    with open(path, "rb") as f:
        st = pickle.load(f)
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def env_ssl():
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        os.environ["CURL_CA_BUNDLE"] = certifi.where()
    except Exception:
        pass
    os.environ.pop("HF_HUB_OFFLINE", None)
