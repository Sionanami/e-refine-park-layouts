# -*- coding: utf-8 -*-
"""Stage-1 LoRA trainer (overnight_fixed_v2) - no hidden lr scale, panels must not silently skip."""
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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MODEL,
    PROMPT,
    RESOLUTION,
    SEEDS_PANEL,
    CFG,
    INFER_STEPS,
    diffusion_target,
    dump_rng,
    env_ssl,
    optimizer_lr_snapshot,
    param_norm,
    peak_vram_gb,
    pil_to_b_tensor,
    set_seed,
    trainable_param_vector,
    vector_l2,
    write_json,
)

env_ssl()


class AugBDataset(Dataset):
    """Offline-augmented B only; no online rotate/flip/crop/color."""

    def __init__(self, csv_path: Path, resolution: int = RESOLUTION):
        self.rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.resolution = resolution

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["B_path"]).convert("RGB")
        t = pil_to_b_tensor(img, self.resolution)
        return {
            "pixel_values": t,
            "case_id": r.get("base_case_id") or r.get("case_id"),
            "pair_id": r.get("pair_id", ""),
            "transform_id": r.get("transform_id", "r0"),
        }


def make_warmup_constant(optimizer, warmup_steps: int, base_lr: float):
    def lr_lambda(step: int):
        # step is 0-based call count from scheduler.step(); after first update step=0
        # We call scheduler.step() after each successful optimizer update, starting from global_s...
        # Use global_step-aware external set; here LambdaLR receives # of scheduler.step calls.
        if warmup_steps <= 0:
            return 1.0
        # after k scheduler steps, current step index = k (1-based effective via max)
        # PyTorch LambdaLR: lr = base_lr * lr_lambda(last_epoch)
        # last_epoch starts -1, first step -> 0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cast_lora_dtype(unet, dtype):
    for n, p in unet.named_parameters():
        if "lora" in n.lower():
            p.data = p.data.to(dtype=dtype)


@torch.no_grad()
def gen_panel_inplace(unet, vae, text_encoder, tokenizer, scheduler, device, out_path: Path, seeds):
    """In-process panel: cast LoRA+VAE to fp16 for pipeline; restore train dtypes. Raises on failure."""
    cast_lora_dtype(unet, torch.float16)
    unet.eval()
    vae_was = next(vae.parameters()).dtype
    te_dev = next(text_encoder.parameters()).device
    text_encoder.to(device)
    vae.to(dtype=torch.float16)
    try:
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        imgs = []
        for s in seeds:
            g = torch.Generator(device=device).manual_seed(s)
            imgs.append(
                pipe(PROMPT, num_inference_steps=INFER_STEPS, generator=g, guidance_scale=CFG).images[0]
            )
        w, h = imgs[0].size
        canvas = Image.new("RGB", (w * len(imgs) + 4 * (len(imgs) - 1), h), (180, 180, 180))
        x = 0
        for im in imgs:
            canvas.paste(im, (x, 0))
            x += w + 4
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        del pipe
    finally:
        vae.to(dtype=vae_was)
        cast_lora_dtype(unet, torch.float32)
        for n, p in unet.named_parameters():
            if "lora" in n.lower() and p.requires_grad:
                p.data = p.data.float()
        text_encoder.to(te_dev)
        unet.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_full_ckpt(path: Path, unet, optimizer, scheduler, scaler, global_step, meta: dict):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(unet), path / "pytorch_lora_weights.bin")
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
    write_json(path / "lora_meta.json", meta)
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
    for name in ("pytorch_lora_weights.bin", "train_state.pt"):
        fp = path / name
        if fp.exists():
            sha[name] = _sha256_file(fp)
    write_json(path / "SHA256.json", sha)


def run_fixed_denoise_eval(unet, vae, text_emb, noise_scheduler, device, pack: dict) -> float:
    """Fixed samples/timesteps/noise; no weight update. Returns mean MSE."""
    unet.eval()
    losses = []
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        for item in pack["items"]:
            pv = item["pixel_values"].to(device, dtype=vae_dtype)
            latents = vae.encode(pv).latent_dist.mean * vae.config.scaling_factor
            latents = latents.to(dtype=torch.float16)
            noise = item["noise"].to(device, dtype=latents.dtype)
            timesteps = item["timesteps"].to(device)
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            target, _ = diffusion_target(noise_scheduler, noise, latents, timesteps)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                pred = unet(noisy, timesteps, text_emb.expand(latents.shape[0], -1, -1), return_dict=False)[0]
            losses.append(float(F.mse_loss(pred.float(), target.float()).cpu()))
    unet.train()
    return float(np.mean(losses)) if losses else float("nan")


def build_fixed_eval_pack(csv_path: Path, vae, device, out_path: Path, n: int = 8, seed: int = 42):
    """Pre-save fixed eval tensors (B images from val or train subset)."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))[:n]
    g = torch.Generator(device="cpu").manual_seed(seed)
    items = []
    meta_items = []
    for r in rows:
        img = Image.open(r["B_path"]).convert("RGB")
        pv = pil_to_b_tensor(img, RESOLUTION).unsqueeze(0)
        with torch.no_grad():
            vae_dtype = next(vae.parameters()).dtype
            lat = vae.encode(pv.to(device, dtype=vae_dtype)).latent_dist.mean * vae.config.scaling_factor
        noise = torch.randn(lat.shape, generator=g, dtype=torch.float16)
        # fixed timesteps
        ts = torch.tensor([200, 500, 800], dtype=torch.long)
        for t in ts:
            items.append(
                {
                    "pixel_values": pv.clone(),
                    "noise": noise.clone(),
                    "timesteps": t.view(1),
                }
            )
            meta_items.append(
                {
                    "case": r.get("base_case_id") or r.get("case_id"),
                    "timestep": int(t),
                    "noise_sha": hashlib_tensor(noise),
                }
            )
    pack = {"items": items, "meta": {"seed": seed, "items": meta_items, "latent": "posterior_mean"}}
    torch.save({"meta": pack["meta"], "cpu_items": items}, out_path)
    return pack


def hashlib_tensor(t: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--val_manifest_csv", required=True)
    ap.add_argument("--fixed_eval_pack", required=True)
    ap.add_argument("--max_train_steps", type=int, required=True)
    ap.add_argument("--save_every_steps", type=int, default=100)
    ap.add_argument("--panel_every_steps", type=int, default=200)
    ap.add_argument("--ckpt_every_sec", type=float, default=300.0)
    ap.add_argument("--deadline_mono", type=float, default=0.0, help="time.monotonic deadline; 0=ignore")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from out_dir/checkpoint-latest if present",
    )
    ap.add_argument(
        "--skip_panels",
        action="store_true",
        help="Skip in-process panels; orchestrator runs panel_worker after exit",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = out / "panels"
    panels.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(str(MODEL), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(str(MODEL), subfolder="text_encoder").to(
        device, dtype=torch.float16
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    # fp16 VAE on 6GB WDDM (fp32 VAE + resume AdamW previously tipped into OOM/CUBLAS fail)
    vae = AutoencoderKL.from_pretrained(
        str(MODEL), subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    unet = UNet2DConditionModel.from_pretrained(
        str(MODEL), subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )
    unet.requires_grad_(False)
    unet.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(str(MODEL), subfolder="scheduler")
    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")

    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    for n, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            p.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    try:
        unet.set_attn_processor  # noqa: B018
        from diffusers.models.attention_processor import AttnProcessor2_0

        unet.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    assert lora_params and all(p.dtype == torch.float32 for p in lora_params)

    # foreach=True can spike VRAM on resume (WDDM 6GB); keep fused/foreach off for stability
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )
    # Explicit: no scale_lr
    scale_lr = False
    assert abs(optimizer.param_groups[0]["lr"] - args.lr) < 1e-15
    lr_sched = make_warmup_constant(optimizer, args.warmup_steps, args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_step = 0
    resumed = False
    resume_dir = out / "checkpoint-latest"
    if args.resume and (resume_dir / "pytorch_lora_weights.bin").exists():
        set_peft_model_state_dict(unet, torch.load(resume_dir / "pytorch_lora_weights.bin", map_location="cpu"))
        for n, p in unet.named_parameters():
            if "lora" in n.lower() and p.requires_grad:
                p.data = p.data.float()
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
        rng_path = resume_dir / "rng.pkl"
        if rng_path.exists():
            try:
                from common import load_rng

                load_rng(rng_path)
            except Exception:
                pass
        resumed = True
        write_json(out / "RESUMED_FROM.json", {"dir": str(resume_dir), "start_step": start_step})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with torch.no_grad():
        tok = tokenizer(
            [PROMPT],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_emb = text_encoder(tok.input_ids.to(device))[0]
    # Offload text encoder when panels are deferred (saves ~0.3GB on 6GB WDDM)
    if args.skip_panels:
        text_encoder.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # fixed eval pack
    fep = Path(args.fixed_eval_pack)
    if fep.exists():
        blob = torch.load(fep, map_location="cpu")
        pack = {"items": blob["cpu_items"], "meta": blob["meta"]}
    else:
        pack = build_fixed_eval_pack(Path(args.val_manifest_csv), vae, device, fep, n=8, seed=42)

    ds = AugBDataset(Path(args.manifest_csv), RESOLUTION)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    data_iter = iter(dl)

    opt_snap0 = optimizer_lr_snapshot(optimizer)
    vec0 = trainable_param_vector(unet, "lora")
    # freeze check baseline for non-lora
    frozen_sum0 = sum(
        float(p.detach().float().abs().sum().cpu())
        for n, p in unet.named_parameters()
        if "lora" not in n.lower()
    )

    cfg = {
        "stage": "lora",
        "model": str(MODEL),
        "prediction_type": pred_type,
        "rank": args.rank,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lr_requested": args.lr,
        "scale_lr": scale_lr,
        "optimizer": opt_snap0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "batch": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": 1.0,
        "amp": "fp16_autocast+GradScaler",
        "trainable": "LoRA only; VAE/text_encoder/UNet base frozen",
        "online_aug": "none",
        "manifest_csv": str(args.manifest_csv),
        "n_train_pairs": len(ds),
        "seed": args.seed,
        "max_train_steps": args.max_train_steps,
        "global_step_definition": "successful optimizer updates (finite grads and scaler.step applied)",
        "preflight": bool(args.preflight),
    }
    write_json(out / "EFFECTIVE_CONFIG_STAGE1.json", cfg)
    write_json(out / "optimizer_lr_at_start.json", opt_snap0)

    loss_csv = out / "train_loss.csv"
    eval_csv = out / "fixed_denoise_eval.csv"
    if not resumed or not loss_csv.exists():
        loss_csv.write_text("step,loss,lr,peak_vram_gb,scaler_skips,sec\n", encoding="utf-8")
    if not resumed or not eval_csv.exists():
        eval_csv.write_text("step,fixed_denoise_mse\n", encoding="utf-8")

    best_path = out / "checkpoint-best" / "lora_meta.json"
    if resumed and best_path.exists():
        try:
            bm = json.loads(best_path.read_text(encoding="utf-8"))
            best = {"step": int(bm.get("step") or start_step), "mse": float(bm.get("fixed_denoise_mse") or 1e9)}
        except Exception:
            best = {"step": start_step, "mse": 1e9}
    else:
        mse0 = run_fixed_denoise_eval(unet, vae, text_emb, noise_scheduler, device, pack)
        with eval_csv.open("a", encoding="utf-8") as f:
            f.write(f"0,{mse0}\n")
        best = {"step": 0, "mse": mse0}
        save_full_ckpt(
            out / "checkpoint-0",
            unet,
            optimizer,
            lr_sched,
            scaler,
            0,
            {"step": 0, "fixed_denoise_mse": mse0, **cfg},
        )
        if args.skip_panels:
            write_json(
                out / "PANEL_DEFERRED.json",
                {"deferred": True, "reason": "orchestrator panel_worker after trainer exit"},
            )
            text_encoder.to("cpu")
        else:
            try:
                gen_panel_inplace(
                    unet, vae, text_encoder, tokenizer, noise_scheduler, device, panels / "step0.png", SEEDS_PANEL
                )
                text_encoder.to("cpu")
            except Exception as e:
                write_json(out / "PANEL_ERROR.json", {"step": 0, "error": repr(e)})
                raise RuntimeError(f"panel generation failed at step0 (must not silently skip): {e}") from e
    text_encoder.to("cpu")
    panel_ok = True

    status = {
        "phase": "lora_train",
        "global_step": start_step,
        "best": best,
        "panel_ok": panel_ok,
        "resumed": resumed,
        "scaler_skips": 0,
        "started_mono": time.monotonic(),
        "started_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(out / "RUN_STATUS_STAGE1.json", status)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_step = start_step
    micro = 0
    scaler_skips = 0
    successful_samples = start_step * args.gradient_accumulation_steps
    t_last_ckpt = time.monotonic()
    nan = False
    stop_reason = None
    optimizer.zero_grad(set_to_none=True)
    log_steps = []

    try:
        while global_step < args.max_train_steps:
            if args.deadline_mono and time.monotonic() >= args.deadline_mono:
                stop_reason = "deadline"
                status["stop_reason"] = stop_reason
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            t0 = time.time()
            pv = batch["pixel_values"].to(device, dtype=torch.float16)
            with torch.no_grad():
                # sample posterior (documented); VAE kept fp16 for VRAM
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float16)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            target, _ = diffusion_target(noise_scheduler, noise, latents, timesteps)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                model_pred = unet(noisy, timesteps, text_emb.expand(bsz, -1, -1), return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # scale loss once for grad accum
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            micro += 1

            if micro % args.gradient_accumulation_steps != 0:
                continue

            scaler.unscale_(optimizer)
            grads_ok = True
            for p in lora_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False
                    nan = True
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)

            scale_before = scaler.get_scale()
            stepped = False
            if grads_ok:
                scaler.step(optimizer)
                stepped = True
            else:
                scaler_skips += 1
            scaler.update()
            scale_after = scaler.get_scale()
            # GradScaler skipped step if scale decreased (inf detected internally)
            if stepped and scale_after < scale_before:
                scaler_skips += 1
                stepped = False

            optimizer.zero_grad(set_to_none=True)

            if not stepped:
                if nan:
                    break
                continue

            # successful optimizer update
            lr_sched.step()
            global_step += 1
            successful_samples += args.gradient_accumulation_steps
            lr_now = float(optimizer.param_groups[0]["lr"])
            row = {
                "step": global_step,
                "loss": float(loss.detach().float().cpu()),
                "lr": lr_now,
                "sec": time.time() - t0,
                "peak_vram_gb": peak_vram_gb(),
                "scaler_skips": scaler_skips,
                "lora_norm": param_norm(unet, "lora"),
            }
            log_steps.append(row)
            print(row, flush=True)
            with loss_csv.open("a", encoding="utf-8") as f:
                f.write(
                    f"{global_step},{row['loss']},{lr_now},{row['peak_vram_gb']},{scaler_skips},{row['sec']}\n"
                )

            # periodic fixed eval
            do_eval = global_step % args.save_every_steps == 0 or global_step == args.max_train_steps
            if do_eval:
                mse = run_fixed_denoise_eval(unet, vae, text_emb, noise_scheduler, device, pack)
                with eval_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{global_step},{mse}\n")
                if math.isfinite(mse) and (mse < best["mse"] or (mse == best["mse"] and global_step < best["step"])):
                    best = {"step": global_step, "mse": mse}
                    save_full_ckpt(
                        out / "checkpoint-best",
                        unet,
                        optimizer,
                        lr_sched,
                        scaler,
                        global_step,
                        {"step": global_step, "fixed_denoise_mse": mse, "selection": "min_fixed_denoise", **cfg},
                    )

            # save latest periodically / by time
            need_ckpt = (
                do_eval
                or (time.monotonic() - t_last_ckpt) >= args.ckpt_every_sec
                or global_step == args.max_train_steps
            )
            if need_ckpt:
                save_full_ckpt(
                    out / "checkpoint-latest",
                    unet,
                    optimizer,
                    lr_sched,
                    scaler,
                    global_step,
                    {"step": global_step, "fixed_denoise_mse": best.get("mse"), **cfg},
                )
                save_full_ckpt(
                    out / f"checkpoint-{global_step}",
                    unet,
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
                    gen_panel_inplace(
                        unet,
                        vae,
                        text_encoder,
                        tokenizer,
                        noise_scheduler,
                        device,
                        panels / f"step{global_step}.png",
                        SEEDS_PANEL,
                    )
                    text_encoder.to("cpu")
                except Exception as e:
                    write_json(out / "PANEL_ERROR.json", {"step": global_step, "error": repr(e)})
                    raise RuntimeError(f"panel generation failed at step {global_step}: {e}") from e

            status.update(
                {
                    "global_step": global_step,
                    "best": best,
                    "scaler_skips": scaler_skips,
                    "successful_sample_presentations": successful_samples,
                    "optimizer": optimizer_lr_snapshot(optimizer),
                    "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            write_json(out / "RUN_STATUS_STAGE1.json", status)
            write_json(
                out / "STAGE1_SUMMARY.json",
                {
                    "completed_steps": global_step,
                    "best": best,
                    "scaler_skips": scaler_skips,
                    "successful_sample_presentations": successful_samples,
                    "optimizer_final": status.get("optimizer"),
                    "nan": nan,
                    "peak_vram_gb": peak_vram_gb(),
                    "phase": "lora_running_or_done",
                    "resumed": resumed,
                },
            )

            if nan:
                stop_reason = "nan"
                break
    except torch.cuda.OutOfMemoryError as e:
        write_json(out / "OOM.json", {"oom": True, "error": str(e), "peak_vram_gb": peak_vram_gb()})
        raise SystemExit(76)

    # Final save before exit (especially important on deadline / crash-prone hosts)
    try:
        save_full_ckpt(
            out / "checkpoint-latest",
            unet,
            optimizer,
            lr_sched,
            scaler,
            global_step,
            {"step": global_step, "stop_reason": stop_reason, **cfg},
        )
    except Exception as e:
        write_json(out / "FINAL_SAVE_WARN.json", {"error": repr(e)})

    summary = {
        "completed_steps": global_step,
        "max_train_steps": args.max_train_steps,
        "best": best,
        "scaler_skips": scaler_skips,
        "successful_sample_presentations": successful_samples,
        "optimizer_final": optimizer_lr_snapshot(optimizer),
        "nan": nan,
        "peak_vram_gb": peak_vram_gb(),
        "steps_tail": log_steps[-5:],
        "stop_reason": stop_reason,
        "resumed": resumed,
        "phase": "lora_done",
    }
    try:
        summary["lora_weight_delta_l2"] = vector_l2(vec0, trainable_param_vector(unet, "lora"))
    except Exception as e:
        summary["delta_error"] = repr(e)
    write_json(out / "STAGE1_SUMMARY.json", summary)
    write_json(out / "RUN_STATUS_STAGE1.json", {**status, **summary})
    print("DONE_LORA", json.dumps(summary, indent=2), flush=True)
    # Success if we made progress or hit deadline with any steps (incl. resumed)
    code = 0 if (global_step > 0 and not nan) or (stop_reason == "deadline" and global_step >= start_step) else 2
    if stop_reason == "deadline" and global_step > 0:
        code = 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
