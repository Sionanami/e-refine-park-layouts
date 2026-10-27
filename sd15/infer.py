# -*- coding: utf-8 -*-
"""SD1.5 + LoRA + T2I-Adapter inference."""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.model:
        os.environ["SD15_MODEL"] = str(args.model)

    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionAdapterPipeline, T2IAdapter, UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTokenizer

    from common import CFG, MODEL, PROMPT, RESOLUTION, env_ssl
    from hard_constraint_compose import hard_constraint_compose
    from site_compose import compose_Y
    from site_target import load_building_area_budget, paste_forbidden_from_A, sanitize_condition_A

    env_ssl()
    device = torch.device("cuda:0")
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)

    unet_dir = MODEL / "unet"
    fp16 = unet_dir / "diffusion_pytorch_model.fp16.safetensors"
    weight_path = fp16 if fp16.exists() else unet_dir / "diffusion_pytorch_model.safetensors"
    sd = load_file(str(weight_path), device="cpu")
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(str(unet_dir)))
    unet.load_state_dict(sd, strict=False)
    del sd
    gc.collect()
    unet.to(dtype=torch.float16)
    unet.add_adapter(LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    set_peft_model_state_dict(unet, torch.load(args.lora, map_location="cpu", weights_only=True))
    unet.requires_grad_(False)
    unet.eval()
    unet.to(device=device, dtype=torch.float16)

    tokenizer = CLIPTokenizer.from_pretrained(str(MODEL), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(str(MODEL), subfolder="text_encoder").to(device, dtype=torch.float16)
    text_encoder.eval()
    vae = AutoencoderKL.from_pretrained(str(MODEL), subfolder="vae", torch_dtype=torch.float16).to(device)
    vae.eval()
    scheduler = DDPMScheduler.from_pretrained(str(MODEL), subfolder="scheduler")
    adapter = T2IAdapter.from_pretrained(str(args.adapter))
    adapter.to(device=device, dtype=torch.float16)
    adapter.eval()
    pipe = StableDiffusionAdapterPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        adapter=adapter, scheduler=scheduler, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    bcap = float(load_building_area_budget()["max_hard"])

    for r in rows:
        cid = str(r.get("base_case_id") or r.get("case_id"))
        case_dir = args.out / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        a = Image.open(r["A_path"]).convert("RGB")
        b = Image.open(r["B_path"]).convert("RGB")
        if a.size != (RESOLUTION, RESOLUTION):
            a = a.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
            b = b.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
        a_s = sanitize_condition_A(a)
        a.save(case_dir / "A.png")
        b.save(case_dir / "B.png")
        cond = torch.from_numpy(np.array(a_s).astype("float32").transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        g = torch.Generator(device=device).manual_seed(int(args.seed))
        with torch.inference_mode():
            raw = pipe(
                prompt=PROMPT, image=cond, num_inference_steps=int(args.steps),
                generator=g, guidance_scale=CFG, adapter_conditioning_scale=1.15,
            ).images[0]
        a_np = np.array(a)
        raw_np = np.array(raw.convert("RGB"))
        if raw_np.shape[:2] != a_np.shape[:2]:
            raw_np = np.array(Image.fromarray(raw_np).resize((a_np.shape[1], a_np.shape[0]), Image.NEAREST))
        Image.fromarray(raw_np).save(case_dir / "rawG_free.png")
        raw_np = paste_forbidden_from_A(raw_np, a_np, urban_dilate=1)
        raw_np, _ = hard_constraint_compose(
            raw_np, a_np, building_max_frac=bcap, preserve_geometry=True,
            road_as_centerline=False, plant_as_disks=False, road_width=1,
        )
        Image.fromarray(raw_np).save(case_dir / "rawG.png")
        y_comp, _ = compose_Y(raw_np, a_np)
        Image.fromarray(y_comp).save(case_dir / "composeY.png")
        print(f"INFER_OK {cid}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
