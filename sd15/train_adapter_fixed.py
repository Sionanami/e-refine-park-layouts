# -*- coding: utf-8 -*-
"""Stage-2 T2I-Adapter trainer (overnight_fixed_v2) - AdamW8bit only, no SGD fallback."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CFG,
    INFER_STEPS,
    MODEL,
    PROMPT,
    RESOLUTION,
    SEEDS_PANEL,
    diffusion_target,
    dump_rng,
    env_ssl,
    optimizer_lr_snapshot,
    peak_vram_gb,
    pil_to_a_tensor,
    pil_to_b_tensor,
    set_seed,
    trainable_param_vector,
    vector_l2,
    write_json,
)

env_ssl()


class AugABDataset(Dataset):
    def __init__(self, csv_path: Path, resolution: int = RESOLUTION, target_mode: str = "raw_b"):
        self.rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.resolution = resolution
        _ok = {
            "raw_b",
            "site_compose",
            "centerline_compose",
            "fused_centerline_compose",
            "proportion_compose",
            "skeleton_first_compose",
        }
        if target_mode not in _ok:
            raise ValueError(target_mode)
        self.target_mode = target_mode
        self.sanitize_condition = False
        self.composition_condition = False  # site+urban + area-budget legend (implies sanitize)
        self.hard_constraint_target = False  # bake hard_constraint into supervision RGB

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import gc
        from composition_condition import build_condition_A
        from site_target import (
            apply_hard_constraint_to_target,
            compose_train_target,
            compose_centerline_train_target,
            compose_fused_centerline_train_target,
            compose_proportion_train_target,
            compose_skeleton_first_train_target,
            sanitize_condition_A,
        )

        r = self.rows[idx]
        last = None
        for _ in range(3):
            try:
                a = Image.open(r["A_path"]).convert("RGB")
                b = Image.open(r["B_path"]).convert("RGB")
                cid = r.get("base_case_id") or r.get("case_id")
                if self.target_mode == "skeleton_first_compose":
                    target = compose_skeleton_first_train_target(a, b, case_id=str(cid), road_width=2)
                elif self.target_mode == "proportion_compose":
                    # proportion_compose already bakes hard when apply_hard=True
                    target = compose_proportion_train_target(
                        a, b, case_id=str(cid), road_width=2, apply_hard=bool(self.hard_constraint_target)
                    )
                elif self.target_mode == "fused_centerline_compose":
                    target = compose_fused_centerline_train_target(a, b, case_id=str(cid), road_width=2)
                elif self.target_mode == "centerline_compose":
                    target = compose_centerline_train_target(a, b, case_id=str(cid), road_width=2)
                elif self.target_mode == "site_compose":
                    target = compose_train_target(a, b)
                else:
                    target = b
                if self.hard_constraint_target and self.target_mode != "proportion_compose":
                    target = apply_hard_constraint_to_target(target, a)  # building ≤ train p90
                if self.composition_condition:
                    # Train teacher: paint area bars from B fracs inside site
                    cond_img = build_condition_A(a, mode="composition", b_img=b)
                elif self.sanitize_condition:
                    cond_img = sanitize_condition_A(a)
                else:
                    cond_img = a
                return {
                    "conditioning_pixel_values": pil_to_a_tensor(cond_img, self.resolution),
                    "pixel_values": pil_to_b_tensor(target, self.resolution),
                    "case_id": cid,
                    "pair_id": r.get("pair_id", ""),
                }
            except MemoryError as e:
                last = e
                gc.collect()
        raise last


def road_structure_weight(pv: torch.Tensor, latent_h: int, latent_w: int, boost: float = 4.0) -> torch.Tensor:
    """Spatial weight map (B,4,h,w): boost orange road / yellow pavement thin structure pixels.

    Soft-clDice-inspired: put more MSE mass on tubular circulation pixels without decoding.
    pv: BCHW in [-1,1].
    """
    return multi_channel_structure_weight(
        pv, latent_h, latent_w, boost_road=boost, boost_pav=boost * 0.75, boost_water=1.0, boost_building=1.0, boost_plant=1.0
    )


def multi_channel_structure_weight(
    pv: torch.Tensor,
    latent_h: int,
    latent_w: int,
    *,
    boost_road: float = 4.0,
    boost_pav: float = 3.0,
    boost_water: float = 5.0,
    boost_building: float = 5.0,
    boost_plant: float = 4.5,
) -> torch.Tensor:
    """Per-class spatial weights for multi-channel gating loss (latent MSE reweight).

    pv: BCHW in [-1,1] (train target RGB). Classes detected by palette heuristics.
    """
    rgb = (pv.float() + 1.0) * 0.5  # [0,1]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Exact-ish palette heuristics; building = magenta (not purple urban_road)
    road = ((r > 0.70) & (g > 0.30) & (g < 0.85) & (b < 0.55) & (r > g) & (r > b + 0.15)).float()
    pav = ((r > 0.75) & (g > 0.70) & (b < 0.55) & (r > b + 0.2)).float()
    water = ((b > 0.70) & (g > 0.70) & (r < 0.50) & (b > r) & (g > r)).float()
    building = ((r > 0.85) & (b > 0.85) & (g < 0.35)).float()  # (255,0,255) only
    plant = ((g > 0.35) & (g < 0.85) & (r < 0.40) & (b < 0.45) & (g > r + 0.1)).float()
    w = (
        1.0
        + (boost_road - 1.0) * road
        + (boost_pav - 1.0) * pav
        + (boost_water - 1.0) * water
        + (boost_building - 1.0) * building
        + (boost_plant - 1.0) * plant
    )
    w = F.interpolate(w.unsqueeze(1), size=(latent_h, latent_w), mode="area")
    return w.expand(-1, 4, -1, -1)


def soft_cldice_proxy_loss(pred: torch.Tensor, target: torch.Tensor, pv: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Topology proxy on road∪pavement circulation + interaction band (site/elements along network)."""
    rgb = (pv.float() + 1.0) * 0.5
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    road = ((r > 0.70) & (g > 0.30) & (g < 0.85) & (b < 0.55) & (r > g) & (r > b + 0.15)).float()
    pav = ((r > 0.75) & (g > 0.70) & (b < 0.55) & (r > b + 0.2)).float()
    circ = ((road + pav) > 0).float().unsqueeze(1)
    # interaction halo: where buildings/plants/water meet the network
    halo = F.max_pool2d(circ, kernel_size=5, stride=1, padding=2) - circ
    halo = halo.clamp(0, 1)
    w = 1.0 + 6.0 * circ + 3.0 * halo  # strong on network, medium on contact band
    w = F.interpolate(w, size=(pred.shape[-2], pred.shape[-1]), mode="area")
    w = w.expand(-1, 4, -1, -1)
    diff = (pred.float() - target.float()) ** 2
    return (diff * w).sum() / (w.sum() + eps)


def multi_channel_gate_loss(pred: torch.Tensor, target: torch.Tensor, pv: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Multi-channel gated MSE so RAW denoising learns constrained class layout (not post-hoc paint).

    Inverse-frequency: rare classes in this target get higher weight (area budget in learning).
    """
    w = multi_channel_structure_weight(
        pv,
        pred.shape[-2],
        pred.shape[-1],
        boost_road=5.5,
        boost_pav=3.0,
        boost_water=4.5,
        boost_building=7.5,
        boost_plant=5.5,
    )
    rgb = (pv.float() + 1.0) * 0.5
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    road = ((r > 0.70) & (g > 0.30) & (g < 0.85) & (b < 0.55) & (r > g) & (r > b + 0.15)).float()
    pav = ((r > 0.75) & (g > 0.70) & (b < 0.55) & (r > b + 0.2)).float()
    water = ((b > 0.70) & (g > 0.70) & (r < 0.50) & (b > r) & (g > r)).float()
    building = ((r > 0.85) & (b > 0.85) & (g < 0.35)).float()
    plant = ((g > 0.35) & (g < 0.85) & (r < 0.40) & (b < 0.45) & (g > r + 0.1)).float()
    inv = torch.ones_like(w[:, :1])
    for m in (road, pav, water, building, plant):
        frac = m.mean(dim=(1, 2), keepdim=True).clamp_min(0.02)
        scale = (0.08 / frac).clamp(0.5, 4.0)
        inv = inv + (scale - 1.0) * F.interpolate(
            m.unsqueeze(1), size=(pred.shape[-2], pred.shape[-1]), mode="area"
        )
    w = w * inv.expand(-1, 4, -1, -1)
    circ = ((road + pav) > 0).float()
    circ_w = F.interpolate(circ.unsqueeze(1), size=(pred.shape[-2], pred.shape[-1]), mode="area")
    w = w * (1.0 + 0.5 * circ_w.expand(-1, 4, -1, -1))
    diff = (pred.float() - target.float()) ** 2
    return (diff * w).sum() / (w.sum() + eps)


def freeze_region_weight_from_cond(
    cond: torch.Tensor,
    latent_h: int,
    latent_w: int,
    *,
    boost: float = 8.0,
) -> torch.Tensor:
    """禁生区高权：场外 + 紫城路 + 红线（来自 sanitize 条件图）。

    仅场地内部绿色可学 building/water/plant/road/pavement。
    """
    r, g, b = cond[:, 0], cond[:, 1], cond[:, 2]
    site = (g > 0.55) & (g > r + 0.12) & (g > b + 0.12)
    # purple urban_road (exclude redline / magenta)
    urban = (r > 0.35) & (b > 0.40) & (g < 0.55) & (r + b > 0.70) & (g < r * 0.90) & (b > g) & ~site
    # redline: high R, low G/B
    redline = (r > 0.65) & (g < 0.45) & (b < 0.45) & (r > g + 0.20) & (r > b + 0.20)
    freeze = (~site) | urban | redline
    # Dilate urban/redline slightly so building magenta cannot bleed onto roads
    freeze_f = F.max_pool2d(freeze.float().unsqueeze(1), kernel_size=3, stride=1, padding=1)
    freeze = freeze_f.squeeze(1) > 0.5
    w = 1.0 + (boost - 1.0) * freeze.float()
    w = F.interpolate(w.unsqueeze(1), size=(latent_h, latent_w), mode="nearest")
    return w.expand(-1, 4, -1, -1)


def freeze_region_mse(
    pred: torch.Tensor, target: torch.Tensor, cond: torch.Tensor, boost: float = 8.0, eps: float = 1e-5
) -> torch.Tensor:
    """Structure-preserving MSE with freeze boost on 禁生区 (outside / urban / redline)."""
    w = freeze_region_weight_from_cond(cond, pred.shape[-2], pred.shape[-1], boost=boost)
    diff = (pred.float() - target.float()) ** 2
    return (diff * w).sum() / (w.sum() + eps)


def urban_building_sep_loss(
    pred: torch.Tensor, target: torch.Tensor, cond: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Extra penalty: 城路/红线/场外必须贴 target(=A)，禁止被学成建筑洋红。"""
    r, g, b = cond[:, 0], cond[:, 1], cond[:, 2]
    site = (g > 0.55) & (g > r + 0.12) & (g > b + 0.12)
    urban = (r > 0.35) & (b > 0.40) & (g < 0.55) & (r + b > 0.70) & ~site
    redline = (r > 0.65) & (g < 0.45) & (b < 0.45) & (r > g + 0.20)
    forbid = ((~site) | urban | redline).float().unsqueeze(1)
    forbid = F.max_pool2d(forbid, kernel_size=3, stride=1, padding=1)
    forbid = F.interpolate(forbid, size=(pred.shape[-2], pred.shape[-1]), mode="nearest")
    diff = (pred.float() - target.float()) ** 2
    return (diff * forbid).sum() / (forbid.sum() * pred.shape[1] + eps)


def make_warmup_constant(optimizer, warmup_steps: int):
    def lr_lambda(step: int):
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_unet_frozen_lora(device, lora_bin: Path):
    """WDDM 6GB-safe load: config+safetensors on CPU, LoRA, then one-shot .to(cuda).

    Avoid from_pretrained(fp16) under low commit/pagefile — it peaks higher and
    frequently raises 0xC0000005 / WinError 1455 on this host.
    """
    import gc

    from safetensors.torch import load_file

    unet_dir = MODEL / "unet"
    fp16 = unet_dir / "diffusion_pytorch_model.fp16.safetensors"
    fp32 = unet_dir / "diffusion_pytorch_model.safetensors"
    weight_path = fp16 if fp16.exists() else fp32
    print(f"unet_weights {weight_path.name}", flush=True)
    sd = load_file(str(weight_path), device="cpu")
    cfg = UNet2DConditionModel.load_config(str(unet_dir))
    unet = UNet2DConditionModel.from_config(cfg)
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    del sd
    gc.collect()
    if missing:
        print(f"unet_missing_keys {len(missing)}", flush=True)
    if unexpected:
        print(f"unet_unexpected_keys {len(unexpected)}", flush=True)
    unet.to(dtype=torch.float16)
    unet.add_adapter(
        LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    state = torch.load(lora_bin, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(unet, state)
    del state
    gc.collect()
    unet.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0

        unet.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass
    unet.eval()
    print("unet_cpu_ready", flush=True)
    if device.type == "cuda":
        gc.collect()
        unet.to(device=device, dtype=torch.float16)
        torch.cuda.empty_cache()
        print("unet_cuda_ready", flush=True)
    else:
        unet.to(device=device, dtype=torch.float16)
    return unet


def param_checksum(module) -> float:
    return float(sum(p.detach().float().abs().sum().cpu() for p in module.parameters()))


def save_adapter_ckpt(path: Path, adapter, optimizer, scheduler, scaler, global_step, meta: dict):
    path.mkdir(parents=True, exist_ok=True)
    adapter.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "global_step": global_step,
            "meta": meta,
        },
        path / "train_state.pt",
    )
    dump_rng(path / "rng.pkl")
    write_json(path / "train_meta.json", meta)
    import hashlib

    def _sha256_file(fp: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    sha = {}
    for p in path.iterdir():
        if not p.is_file():
            continue
        if p.name in {"train_meta.json", "SHA256.json", "rng.pkl"}:
            continue
        if p.suffix in {".bin", ".safetensors", ".pt", ".json"}:
            try:
                sha[p.name] = _sha256_file(p)
            except Exception as e:
                sha[p.name] = f"hash_failed:{e!r}"
    write_json(path / "SHA256.json", sha)


@torch.no_grad()
def panel_adapter(adapter, unet, vae, text_encoder, tokenizer, scheduler, device, a_path: Path, out_path: Path, seed: int):
    from diffusers import StableDiffusionAdapterPipeline

    a = Image.open(a_path).convert("RGB")
    if a.size != (RESOLUTION, RESOLUTION):
        a = a.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
    t = pil_to_a_tensor(a, RESOLUTION).unsqueeze(0).to(device)
    # adapters/pipeline expect fp16-compatible
    ad = adapter
    ad_was_training = ad.training
    ad.eval()
    pipe = StableDiffusionAdapterPipeline(
        vae=vae.to(dtype=torch.float16),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        adapter=ad.to(dtype=torch.float16),
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator(device=device).manual_seed(seed)
    img = pipe(
        prompt=PROMPT,
        image=t,
        num_inference_steps=INFER_STEPS,
        generator=g,
        guidance_scale=CFG,
        adapter_conditioning_scale=1.0,
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    del pipe
    ad.to(dtype=torch.float32)
    vae.to(dtype=torch.float32)
    if ad_was_training:
        ad.train()


def run_fixed_denoise_eval(adapter, unet, vae, text_emb, noise_scheduler, device, pack: dict) -> float:
    adapter.eval()
    losses = []
    with torch.no_grad():
        for item in pack["items"]:
            cond = item["cond"].to(device, dtype=torch.float32)
            pv = item["pixel_values"].to(device, dtype=torch.float32)
            latents = vae.encode(pv).latent_dist.mean * vae.config.scaling_factor
            latents = latents.to(dtype=torch.float16)
            noise = item["noise"].to(device, dtype=latents.dtype)
            timesteps = item["timesteps"].to(device)
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            target, _ = diffusion_target(noise_scheduler, noise, latents, timesteps)
            feats = adapter(cond)
            residuals = [s.to(dtype=torch.float16) for s in feats]
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                pred = unet(
                    noisy,
                    timesteps,
                    text_emb.expand(latents.shape[0], -1, -1),
                    down_intrablock_additional_residuals=residuals,
                    return_dict=False,
                )[0]
            losses.append(float(F.mse_loss(pred.float(), target.float()).cpu()))
    adapter.train()
    return float(np.mean(losses)) if losses else float("nan")


def build_fixed_eval_pack_ab(
    csv_path: Path,
    out_path: Path,
    n: int = 8,
    seed: int = 42,
    target_mode: str = "raw_b",
    sanitize_condition: bool = False,
    composition_condition: bool = False,
    hard_constraint_target: bool = False,
):
    from composition_condition import build_condition_A
    from site_target import (
        apply_hard_constraint_to_target,
        compose_train_target,
        compose_centerline_train_target,
        compose_fused_centerline_train_target,
        compose_proportion_train_target,
        compose_skeleton_first_train_target,
        sanitize_condition_A,
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))[:n]
    g = torch.Generator(device="cpu").manual_seed(seed)
    items = []
    meta = []
    for r in rows:
        a = Image.open(r["A_path"]).convert("RGB")
        b = Image.open(r["B_path"]).convert("RGB")
        cid = r.get("base_case_id") or r.get("case_id")
        if target_mode == "skeleton_first_compose":
            target = compose_skeleton_first_train_target(a, b, case_id=str(cid), road_width=2)
        elif target_mode == "proportion_compose":
            target = compose_proportion_train_target(
                a, b, case_id=str(cid), road_width=2, apply_hard=bool(hard_constraint_target)
            )
        elif target_mode == "fused_centerline_compose":
            target = compose_fused_centerline_train_target(a, b, case_id=str(cid), road_width=2)
        elif target_mode == "centerline_compose":
            target = compose_centerline_train_target(a, b, case_id=str(cid), road_width=2)
        elif target_mode == "site_compose":
            target = compose_train_target(a, b)
        else:
            target = b
        if hard_constraint_target and target_mode != "proportion_compose":
            target = apply_hard_constraint_to_target(target, a)  # building ≤ train p90
        if composition_condition:
            # Val pack: use global p50 bars (no B leakage into condition)
            cond_img = build_condition_A(a, mode="composition", b_img=None)
        elif sanitize_condition:
            cond_img = sanitize_condition_A(a)
        else:
            cond_img = a
        cond = pil_to_a_tensor(cond_img, RESOLUTION).unsqueeze(0)
        pv = pil_to_b_tensor(target, RESOLUTION).unsqueeze(0)
        # latent shape for 512 -> 64x64x4
        noise = torch.randn(1, 4, RESOLUTION // 8, RESOLUTION // 8, generator=g, dtype=torch.float16)
        for t in [200, 500, 800]:
            items.append(
                {
                    "cond": cond.clone(),
                    "pixel_values": pv.clone(),
                    "noise": noise.clone(),
                    "timesteps": torch.tensor([t], dtype=torch.long),
                }
            )
            meta.append({"case": r.get("base_case_id") or r.get("case_id"), "timestep": t})
    blob = {
        "meta": {
            "seed": seed,
            "items": meta,
            "latent_at_eval": "posterior_mean",
            "target_mode": target_mode,
            "sanitize_condition": bool(sanitize_condition or composition_condition),
            "composition_condition": bool(composition_condition),
            "hard_constraint_target": bool(hard_constraint_target),
        },
        "cpu_items": items,
    }
    torch.save(blob, out_path)
    return {"items": items, "meta": blob["meta"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--val_manifest_csv", required=True)
    ap.add_argument("--fixed_eval_pack", required=True)
    ap.add_argument("--lora_weights", required=True)
    ap.add_argument("--max_train_steps", type=int, required=True)
    ap.add_argument("--save_every_steps", type=int, default=100)
    ap.add_argument("--panel_every_steps", type=int, default=200)
    ap.add_argument("--ckpt_every_sec", type=float, default=300.0)
    ap.add_argument("--deadline_mono", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--panel_a_path", type=str, default="")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from out_dir/adapter-latest")
    ap.add_argument("--init_adapter_dir", type=str, default="", help="Load adapter weights only; fresh optimizer, step 0")
    ap.add_argument(
        "--target_mode",
        choices=[
            "raw_b",
            "site_compose",
            "centerline_compose",
            "fused_centerline_compose",
            "proportion_compose",
            "skeleton_first_compose",
        ],
        default="raw_b",
    )
    ap.add_argument(
        "--road_struct_weight",
        type=float,
        default=0.0,
        help="If >0, mix soft-clDice-proxy (structure-weighted MSE) with epsilon MSE; e.g. 0.35",
    )
    ap.add_argument(
        "--multi_channel_weight",
        type=float,
        default=0.0,
        help="If >0, mix multi-channel gated MSE (water/building/plant/road/pav) with epsilon MSE",
    )
    ap.add_argument(
        "--sanitize_condition",
        action="store_true",
        help="Use crisp site+urban_road A as Adapter condition (raw-opt)",
    )
    ap.add_argument(
        "--composition_condition",
        action="store_true",
        help=(
            "DEPRECATED bars removed: same as --sanitize_condition "
            "(site green + urban purple + redline; no bottom legend)"
        ),
    )
    ap.add_argument(
        "--freeze_region_weight",
        type=float,
        default=0.0,
        help="If >0, freeze-boosted MSE on 禁生区 (outside + urban_road + redline); "
        "also enables urban/building separation term",
    )
    ap.add_argument(
        "--hard_constraint_target",
        action="store_true",
        help="Bake geometry-preserving hard_constraint into supervision target (area caps + z-order; keeps road/plant width)",
    )
    ap.add_argument(
        "--skip_panels",
        action="store_true",
        help="Skip in-process panels (recommended on 6GB); orchestrator runs panel_worker after exit",
    )
    args = ap.parse_args()
    if args.composition_condition:
        args.sanitize_condition = True  # composition implies crisp site base

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = out / "panels"
    panels.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # AdamW8bit required - never fall back to SGD
    try:
        import bitsandbytes as bnb
    except Exception as e:
        write_json(
            out / "RESOURCE_BLOCK.json",
            {"blocked": True, "reason": "bitsandbytes_import_failed", "error": repr(e)},
        )
        raise SystemExit(77)

    # Defer init-adapter disk read until AFTER UNet is on CUDA — keeping both in
    # CPU RAM under low commit limit triggers WinError 1455 / 0xC0000005.
    init_adapter_dir = Path(args.init_adapter_dir) if args.init_adapter_dir and not args.resume else None

    tokenizer = CLIPTokenizer.from_pretrained(str(MODEL), subfolder="tokenizer")
    print("loading_unet", flush=True)
    unet = load_unet_frozen_lora(device, Path(args.lora_weights))
    print("unet_ready", flush=True)
    text_encoder = CLIPTextModel.from_pretrained(str(MODEL), subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    vae = AutoencoderKL.from_pretrained(str(MODEL), subfolder="vae", torch_dtype=torch.float32)
    vae.to(device, dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()
    print("vae_ready", flush=True)
    noise_scheduler = DDPMScheduler.from_pretrained(str(MODEL), subfolder="scheduler")
    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    unet_sum0 = param_checksum(unet)
    lora_sum0 = sum(
        float(p.detach().float().abs().sum().cpu()) for n, p in unet.named_parameters() if "lora" in n.lower()
    )

    adapter = T2IAdapter(
        in_channels=3,
        channels=[320, 640, 1280, 1280],
        num_res_blocks=2,
        downscale_factor=8,
        adapter_type="full_adapter",
    )
    if init_adapter_dir is not None:
        st_file = init_adapter_dir / "diffusion_pytorch_model.safetensors"
        bin_file = init_adapter_dir / "diffusion_pytorch_model.bin"
        if st_file.exists():
            from safetensors.torch import load_file

            init_sd = load_file(str(st_file), device="cpu")
        elif bin_file.exists():
            init_sd = torch.load(bin_file, map_location="cpu", weights_only=True)
        else:
            raise SystemExit(f"no adapter weights in {init_adapter_dir}")
        adapter.load_state_dict(init_sd)
        del init_sd
        print("init_adapter_loaded", flush=True)
    adapter.to(device, dtype=torch.float32)
    adapter.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("adapter_ready", flush=True)
    adapter_init = {
        "init": "random_t2i_full_adapter",
        "source": "diffusers.T2IAdapter fresh construction (not pretrained checkpoint)",
        "reason": "No compatible pretrained adapter for this park A/B condition type; avoid mismatched condition init",
        "structure": {
            "adapter_type": "full_adapter",
            "in_channels": 3,
            "channels": [320, 640, 1280, 1280],
            "num_res_blocks": 2,
            "downscale_factor": 8,
        },
        "n_params": sum(p.numel() for p in adapter.parameters()),
        "condition_input": "RGB A in [0,1], NEAREST resize to 512 if needed; sync geom with B offline",
    }
    write_json(out / "ADAPTER_INIT.json", adapter_init)
    adapter_sum0 = param_checksum(adapter)
    vec0 = trainable_param_vector(adapter)

    try:
        optimizer = bnb.optim.AdamW8bit(
            adapter.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
    except Exception as e:
        write_json(
            out / "RESOURCE_BLOCK.json",
            {"blocked": True, "reason": "AdamW8bit_construct_failed", "error": repr(e)},
        )
        raise SystemExit(77)

    # Forbidden: any lr multiplier / SGD
    assert type(optimizer).__name__ == "AdamW8bit"
    assert abs(optimizer.param_groups[0]["lr"] - args.lr) < 1e-15
    lr_sched = make_warmup_constant(optimizer, args.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if args.init_adapter_dir and not args.resume:
        init_dir = Path(args.init_adapter_dir)
        adapter_init["init"] = "weights_from_init_adapter_dir"
        adapter_init["init_adapter_dir"] = str(init_dir)
        adapter_init["optimizer"] = "fresh_AdamW8bit"
        adapter_init["start_step"] = 0
        write_json(out / "ADAPTER_INIT.json", adapter_init)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    start_step = 0
    resumed = False
    resume_dir = out / "adapter-latest"
    if args.resume and resume_dir.exists() and (
        (resume_dir / "diffusion_pytorch_model.safetensors").exists()
        or (resume_dir / "diffusion_pytorch_model.bin").exists()
    ):
        loaded = T2IAdapter.from_pretrained(str(resume_dir))
        adapter.load_state_dict(loaded.state_dict())
        adapter.to(device, dtype=torch.float32)
        adapter.train()
        del loaded
        st_path = resume_dir / "train_state.pt"
        if st_path.exists():
            st = torch.load(st_path, map_location="cpu")
            try:
                optimizer.load_state_dict(st["optimizer"])
            except Exception as e:
                write_json(out / "RESUME_OPT_WARN.json", {"error": repr(e)})
            try:
                if st.get("scheduler") is not None:
                    lr_sched.load_state_dict(st["scheduler"])
            except Exception as e:
                write_json(out / "RESUME_SCHED_WARN.json", {"error": repr(e)})
            try:
                if st.get("scaler") is not None and scaler.is_enabled():
                    scaler.load_state_dict(st["scaler"])
            except Exception as e:
                write_json(out / "RESUME_SCALER_WARN.json", {"error": repr(e)})
            start_step = int(st.get("global_step") or 0)
        resumed = True
        adapter_init["init"] = "resumed_from_adapter_latest"
        adapter_init["resume_step"] = start_step
        write_json(out / "ADAPTER_INIT.json", adapter_init)
        write_json(out / "RESUMED_FROM.json", {"dir": str(resume_dir), "start_step": start_step})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    text_encoder.to(device, dtype=torch.float16)
    with torch.no_grad():
        tok = tokenizer(
            [PROMPT],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_emb = text_encoder(tok.input_ids.to(device))[0]
    # unload text encoder weights from GPU for VRAM; keep CPU copy for panels
    text_encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fep = Path(args.fixed_eval_pack)
    if fep.exists():
        blob = torch.load(fep, map_location="cpu")
        pack = {"items": blob["cpu_items"], "meta": blob["meta"]}
        pack_mode = (blob.get("meta") or {}).get("target_mode", "raw_b")
        pack_san = bool((blob.get("meta") or {}).get("sanitize_condition", False))
        pack_comp = bool((blob.get("meta") or {}).get("composition_condition", False))
        pack_hc = bool((blob.get("meta") or {}).get("hard_constraint_target", False))
        if (
            pack_mode != args.target_mode
            or pack_san != bool(args.sanitize_condition)
            or pack_comp != bool(args.composition_condition)
            or pack_hc != bool(args.hard_constraint_target)
        ):
            raise SystemExit(
                f"eval pack mode={pack_mode}/san={pack_san}/comp={pack_comp}/hc={pack_hc} != "
                f"{args.target_mode}/san={args.sanitize_condition}/comp={args.composition_condition}/hc={args.hard_constraint_target}"
            )
    else:
        pack = build_fixed_eval_pack_ab(
            Path(args.val_manifest_csv),
            fep,
            n=8,
            seed=42,
            target_mode=args.target_mode,
            sanitize_condition=bool(args.sanitize_condition),
            composition_condition=bool(args.composition_condition),
            hard_constraint_target=bool(args.hard_constraint_target),
        )

    ds = AugABDataset(Path(args.manifest_csv), RESOLUTION, target_mode=args.target_mode)
    ds.sanitize_condition = bool(args.sanitize_condition)
    ds.composition_condition = bool(args.composition_condition)
    ds.hard_constraint_target = bool(args.hard_constraint_target)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    data_iter = iter(dl)

    # panel A path
    panel_a = args.panel_a_path
    if not panel_a:
        panel_a = list(csv.DictReader(Path(args.val_manifest_csv).open(encoding="utf-8")))[0]["A_path"]

    cfg = {
        "stage": "adapter",
        "model": str(MODEL),
        "prediction_type": pred_type,
        "frozen_lora": str(args.lora_weights),
        "lr_requested": args.lr,
        "scale_lr": False,
        "optimizer": optimizer_lr_snapshot(optimizer),
        "sgd_fallback_forbidden": True,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "batch": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": 1.0,
        "amp": "fp16_autocast+GradScaler; adapter fp32",
        "unet_no_grad_forbidden": True,
        "adapter_init": adapter_init,
        "manifest_csv": str(args.manifest_csv),
        "target_mode": args.target_mode,
        "target_rule": "Y=M_eff*B+(1-M_eff)*A; city roads and outside site copied from A",
        "road_struct_weight": args.road_struct_weight,
        "multi_channel_weight": args.multi_channel_weight,
        "composition_condition": bool(args.composition_condition),
        "hard_constraint_target": bool(args.hard_constraint_target),
        "freeze_region_weight": float(args.freeze_region_weight or 0.0),
        "condition_encoding": (
            "composition_budget_bars_v1" if args.composition_condition else (
                "sanitize_site_urban_redline" if args.sanitize_condition else "raw_A_rgb"
            )
        ),
        "composition_priors": str(
            Path(__file__).resolve().parent.parent / "manifests" / "COMPOSITION_PRIORS.json"
        ),
        "sanitize_condition": bool(args.sanitize_condition),
        "n_train_pairs": len(ds),
        "seed": args.seed,
        "max_train_steps": args.max_train_steps,
        "global_step_definition": "successful optimizer updates",
        "preflight": bool(args.preflight),
        "memory": [
            "gradient_checkpointing on UNet",
            "SDPA AttnProcessor2_0 if available",
            "text encoder embedding cached then offloaded to CPU",
            "AdamW8bit",
        ],
    }
    write_json(out / "EFFECTIVE_CONFIG_STAGE2.json", cfg)
    write_json(out / "optimizer_lr_at_start.json", optimizer_lr_snapshot(optimizer))

    loss_csv = out / "train_loss.csv"
    eval_csv = out / "fixed_denoise_eval.csv"
    if (not resumed) or (not loss_csv.exists()):
        loss_csv.write_text("step,loss,lr,peak_vram_gb,scaler_skips,adapter_grad_abs,sec\n", encoding="utf-8")
    if (not resumed) or (not eval_csv.exists()):
        eval_csv.write_text("step,fixed_denoise_mse\n", encoding="utf-8")

    best_meta = out / "adapter-best" / "train_meta.json"
    if resumed and best_meta.exists():
        try:
            bm = json.loads(best_meta.read_text(encoding="utf-8"))
            best = {"step": int(bm.get("step") or start_step), "mse": float(bm.get("fixed_denoise_mse") or 1e9)}
        except Exception:
            best = {"step": start_step, "mse": 1e9}
    else:
        mse0 = run_fixed_denoise_eval(adapter, unet, vae, text_emb, noise_scheduler, device, pack)
        with eval_csv.open("a", encoding="utf-8") as f:
            f.write(f"{start_step},{mse0}\n")
        best = {"step": start_step, "mse": mse0}
        save_adapter_ckpt(
            out / "adapter-0",
            adapter,
            optimizer,
            lr_sched,
            scaler,
            start_step,
            {"step": start_step, "fixed_denoise_mse": mse0, **cfg},
        )

    if args.skip_panels:
        write_json(
            out / "PANEL_DEFERRED.json",
            {
                "deferred": True,
                "reason": "in-process panel ACCESS_VIOLATION risk on this host; orchestrator panel_worker after exit",
                "panel_a": panel_a,
            },
        )
    else:
        try:
            text_encoder.to(device)
            panel_adapter(
                adapter,
                unet,
                vae,
                text_encoder,
                tokenizer,
                noise_scheduler,
                device,
                Path(panel_a),
                panels / "step0.png",
                SEEDS_PANEL[0],
            )
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
        except Exception as e:
            write_json(out / "PANEL_ERROR.json", {"step": 0, "error": repr(e)})
            raise RuntimeError(f"adapter panel failed at step0: {e}") from e

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_step = start_step
    micro = 0
    scaler_skips = 0
    successful_samples = start_step * args.gradient_accumulation_steps
    t_last_ckpt = time.monotonic()
    nan = False
    optimizer.zero_grad(set_to_none=True)
    status = {
        "phase": "adapter_train",
        "global_step": global_step,
        "best": best,
        "resumed": resumed,
        "started_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(out / "RUN_STATUS_STAGE2.json", status)

    try:
        while global_step < args.max_train_steps:
            if args.deadline_mono and time.monotonic() >= args.deadline_mono:
                status["stop_reason"] = "deadline"
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            t0 = time.time()
            cond = batch["conditioning_pixel_values"].to(device, dtype=torch.float32)
            pv = batch["pixel_values"].to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float16)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            target, _ = diffusion_target(noise_scheduler, noise, latents, timesteps)
            del latents

            adapter_state = adapter(cond)
            residuals = [s.to(dtype=torch.float16) for s in adapter_state]
            del adapter_state

            # UNet forward WITHOUT no_grad - grads must flow to Adapter
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                model_pred = unet(
                    noisy,
                    timesteps,
                    text_emb.expand(bsz, -1, -1),
                    down_intrablock_additional_residuals=residuals,
                    return_dict=False,
                )[0]
                fr_w = float(getattr(args, "freeze_region_weight", 0.0) or 0.0)
                if fr_w > 0:
                    # 禁生区高权：场外+城路+红线必须贴 target(=A compose)
                    base = freeze_region_mse(model_pred, target, cond, boost=max(8.0, 8.0 + 16.0 * fr_w))
                else:
                    base = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                mc_w = float(getattr(args, "multi_channel_weight", 0.0) or 0.0)
                rs_w = float(args.road_struct_weight or 0.0)
                # base + 五类语义 + 路∪铺拓扑/作用带 + 城路/建筑分离
                mc_w = max(0.0, min(mc_w, 0.70))
                rs_w = max(0.0, min(rs_w, 0.45))  # topology is primary learning focus
                sep_w = 0.15 if fr_w > 0 else 0.0
                mix = mc_w + rs_w + sep_w
                if mix > 0.92:
                    scale = 0.92 / mix
                    mc_w, rs_w, sep_w = mc_w * scale, rs_w * scale, sep_w * scale
                loss = base
                if mc_w > 0 or rs_w > 0 or sep_w > 0:
                    parts = [(1.0 - mc_w - rs_w - sep_w) * base]
                    if mc_w > 0:
                        parts.append(mc_w * multi_channel_gate_loss(model_pred, target, pv))
                    if rs_w > 0:
                        parts.append(rs_w * soft_cldice_proxy_loss(model_pred, target, pv))
                    if sep_w > 0:
                        parts.append(sep_w * urban_building_sep_loss(model_pred, target, cond))
                    loss = parts[0]
                    for p in parts[1:]:
                        loss = loss + p
            del residuals, noisy, model_pred, pv

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            micro += 1
            if micro % args.gradient_accumulation_steps != 0:
                continue

            scaler.unscale_(optimizer)
            grads_finite = True
            grad_abs = 0.0
            for p in adapter.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        grads_finite = False
                        nan = True
                    grad_abs += float(p.grad.detach().float().abs().sum().cpu())
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)

            scale_before = scaler.get_scale()
            stepped = False
            if grads_finite:
                scaler.step(optimizer)
                stepped = True
            else:
                scaler_skips += 1
            scaler.update()
            if stepped and scaler.get_scale() < scale_before:
                scaler_skips += 1
                stepped = False
            optimizer.zero_grad(set_to_none=True)

            if not stepped:
                if nan:
                    break
                continue

            lr_sched.step()
            global_step += 1
            successful_samples += args.gradient_accumulation_steps
            lr_now = float(optimizer.param_groups[0]["lr"])

            unet_sum = param_checksum(unet)
            lora_sum = sum(
                float(p.detach().float().abs().sum().cpu())
                for n, p in unet.named_parameters()
                if "lora" in n.lower()
            )
            row = {
                "step": global_step,
                "loss": float(loss.detach().float().cpu()),
                "lr": lr_now,
                "sec": time.time() - t0,
                "peak_vram_gb": peak_vram_gb(),
                "scaler_skips": scaler_skips,
                "adapter_grad_abs_sum": grad_abs,
                "unet_frozen_ok": abs(unet_sum - unet_sum0) < 1e-2,
                "lora_frozen_ok": abs(lora_sum - lora_sum0) < 1e-2,
            }
            print(row, flush=True)
            with loss_csv.open("a", encoding="utf-8") as f:
                f.write(
                    f"{global_step},{row['loss']},{lr_now},{row['peak_vram_gb']},{scaler_skips},{grad_abs},{row['sec']}\n"
                )

            do_eval = global_step % args.save_every_steps == 0 or global_step == args.max_train_steps
            if do_eval:
                mse = run_fixed_denoise_eval(adapter, unet, vae, text_emb, noise_scheduler, device, pack)
                with eval_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{global_step},{mse}\n")
                if math.isfinite(mse) and (mse < best["mse"] or (mse == best["mse"] and global_step < best["step"])):
                    best = {"step": global_step, "mse": mse}
                    save_adapter_ckpt(
                        out / "adapter-best",
                        adapter,
                        optimizer,
                        lr_sched,
                        scaler,
                        global_step,
                        {"step": global_step, "fixed_denoise_mse": mse, "selection": "min_fixed_denoise", **cfg},
                    )

            need_ckpt = (
                do_eval
                or (time.monotonic() - t_last_ckpt) >= args.ckpt_every_sec
                or global_step == args.max_train_steps
            )
            if need_ckpt:
                save_adapter_ckpt(
                    out / "adapter-latest",
                    adapter,
                    optimizer,
                    lr_sched,
                    scaler,
                    global_step,
                    {"step": global_step, **cfg},
                )
                save_adapter_ckpt(
                    out / f"adapter-{global_step}",
                    adapter,
                    optimizer,
                    lr_sched,
                    scaler,
                    global_step,
                    {"step": global_step, **cfg},
                )
                write_json(out / "optimizer_lr_at_save.json", optimizer_lr_snapshot(optimizer))
                t_last_ckpt = time.monotonic()

            if (not args.skip_panels) and (
                global_step % args.panel_every_steps == 0 or global_step == args.max_train_steps
            ):
                try:
                    text_encoder.to(device)
                    panel_adapter(
                        adapter,
                        unet,
                        vae,
                        text_encoder,
                        tokenizer,
                        noise_scheduler,
                        device,
                        Path(panel_a),
                        panels / f"step{global_step}.png",
                        SEEDS_PANEL[0],
                    )
                    text_encoder.to("cpu")
                    torch.cuda.empty_cache()
                except Exception as e:
                    write_json(out / "PANEL_ERROR.json", {"step": global_step, "error": repr(e)})
                    raise RuntimeError(f"adapter panel failed at step {global_step}: {e}") from e

            status.update(
                {
                    "global_step": global_step,
                    "best": best,
                    "scaler_skips": scaler_skips,
                    "successful_sample_presentations": successful_samples,
                    "optimizer": optimizer_lr_snapshot(optimizer),
                    "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "resumed": resumed,
                }
            )
            write_json(out / "RUN_STATUS_STAGE2.json", status)
            write_json(
                out / "STAGE2_SUMMARY.json",
                {
                    "completed_steps": global_step,
                    "best": best,
                    "scaler_skips": scaler_skips,
                    "successful_sample_presentations": successful_samples,
                    "optimizer_final": status.get("optimizer"),
                    "nan": nan,
                    "peak_vram_gb": peak_vram_gb(),
                    "phase": "adapter_running_or_done",
                    "resumed": resumed,
                },
            )

            if global_step >= args.max_train_steps or nan:
                summary = {
                    "completed_steps": global_step,
                    "best": best,
                    "scaler_skips": scaler_skips,
                    "successful_sample_presentations": successful_samples,
                    "optimizer_final": status.get("optimizer"),
                    "nan": nan,
                    "peak_vram_gb": peak_vram_gb(),
                    "phase": "adapter_done",
                    "exit": "early_os_exit_after_last_step",
                    "resumed": resumed,
                }
                try:
                    summary["adapter_weight_delta_l2"] = vector_l2(vec0, trainable_param_vector(adapter))
                except Exception as e:
                    summary["delta_error"] = repr(e)
                write_json(out / "STAGE2_SUMMARY.json", summary)
                write_json(out / "RUN_STATUS_STAGE2.json", {**status, **summary})
                print("DONE_ADAPTER", json.dumps(summary, indent=2), flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0 if (global_step > 0 and not nan) else 2)
            if nan:
                break
    except torch.cuda.OutOfMemoryError as e:
        write_json(
            out / "RESOURCE_BLOCK.json",
            {
                "blocked": True,
                "reason": "OOM_during_AdamW8bit_adapter",
                "error": str(e),
                "peak_vram_gb": peak_vram_gb(),
                "sgd_fallback": False,
            },
        )
        raise SystemExit(76)

    vec1 = trainable_param_vector(adapter)
    summary = {
        "completed_steps": global_step,
        "best": best,
        "scaler_skips": scaler_skips,
        "successful_sample_presentations": successful_samples,
        "adapter_weight_delta_l2": vector_l2(vec0, vec1),
        "unet_frozen_ok": abs(param_checksum(unet) - unet_sum0) < 1e-2,
        "lora_frozen_ok": abs(
            sum(float(p.detach().float().abs().sum().cpu()) for n, p in unet.named_parameters() if "lora" in n.lower())
            - lora_sum0
        )
        < 1e-2,
        "optimizer_final": optimizer_lr_snapshot(optimizer),
        "nan": nan,
        "peak_vram_gb": peak_vram_gb(),
    }
    write_json(out / "STAGE2_SUMMARY.json", summary)
    write_json(out / "RUN_STATUS_STAGE2.json", {**status, **summary, "phase": "adapter_done"})
    print("DONE_ADAPTER", json.dumps(summary, indent=2), flush=True)
    raise SystemExit(0 if global_step > 0 and not nan else 2)


if __name__ == "__main__":
    main()
