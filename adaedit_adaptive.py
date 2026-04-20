"""
AdaEdit-Adaptive: experimental entry point for controller-driven
injection strength.

This runner does NOT modify the original AdaEdit pipeline. It imports
the original inversion + mask + KV-Mix + Latents-Shift code paths and
only swaps the target-pass sampler when a non-'original' mode is
selected.

Modes:
  - original          : vanilla AdaEdit (identical to adaedit.py).
  - scheduled_fixed   : alias for 'original'. Kept for ablation.
  - fixed_soft        : constant alpha modulation of kv_mix_ratio
                        (no feedback). Useful as a non-adaptive
                        baseline between 'original' and pd_adaptive.
  - pd_adaptive       : PD-controlled alpha driven by drift signal.
  - pid_adaptive      : PID-controlled alpha driven by drift signal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from glob import iglob

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ExifTags

from src.flux.sampling import (
    channel_selective_latents_shift,
    denoise_fireflow,
    get_noise,
    get_schedule,
    latents_shift,
    prepare,
    unpack,
)
from src.flux.util import (
    configs,
    get_word_index,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from src.flux.adaptive import (
    DriftMeter,
    denoise_fireflow_adaptive,
    make_controller,
)

NSFW_THRESHOLD = 0.85
ADAPTIVE_MODES = (
    "original",
    "scheduled_fixed",
    "fixed_soft",
    "pd_adaptive",
    "pid_adaptive",
    "no_injection",
)


@torch.inference_mode()
def _encode_image(image_array, device, ae):
    img = torch.from_numpy(image_array).permute(2, 0, 1).float() / 127.5 - 1
    img = img.unsqueeze(0).to(device)
    return ae.encode(img).to(torch.bfloat16)


def _indices_to_pixel_mask(indices, latent_h, latent_w, img_h, img_w):
    if indices is None or (torch.is_tensor(indices) and indices.numel() == 0):
        return np.zeros((img_h, img_w), dtype=np.float32)
    if not torch.is_tensor(indices):
        return np.zeros((img_h, img_w), dtype=np.float32)
    mask_flat = torch.zeros(latent_h * latent_w, dtype=torch.float32)
    idx = indices.detach().cpu().long()
    idx = idx[(idx >= 0) & (idx < latent_h * latent_w)]
    mask_flat[idx] = 1.0
    mask_2d = mask_flat.view(latent_h, latent_w).numpy()
    pil = Image.fromarray((mask_2d * 255).astype(np.uint8))
    pil = pil.resize((img_w, img_h), resample=Image.NEAREST)
    return (np.array(pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)


def _next_run_dir(root: str, run_name: str) -> str:
    os.makedirs(root, exist_ok=True)
    base = os.path.join(root, run_name)
    if not os.path.exists(base):
        return base
    i = 1
    while os.path.exists(f"{base}_{i:03d}"):
        i += 1
    return f"{base}_{i:03d}"


def run(args):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode not in ADAPTIVE_MODES:
        raise ValueError(
            f"Unknown mode {args.mode!r}. Choose from {ADAPTIVE_MODES}."
        )
    if args.name not in configs:
        raise ValueError(
            f"Unknown model {args.name!r}. Choose from {list(configs.keys())}."
        )

    # --- load models ------------------------------------------------
    print("Loading models...")
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(args.name, device="cpu" if args.offload else device)
    ae = load_ae(args.name, device="cpu" if args.offload else device)
    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(device)

    # --- load source ------------------------------------------------
    print(f"Loading source image: {args.source_img}")
    source_image = np.array(Image.open(args.source_img).convert("RGB"))
    h, w = source_image.shape[:2]
    new_h = h - h % 16
    new_w = w - w % 16
    source_image = source_image[:new_h, :new_w, :]
    print(f"Image size: {new_w}x{new_h}")

    t0 = time.perf_counter()
    init_latent = _encode_image(source_image, device, ae)

    rng = torch.Generator(device="cpu")
    seed = args.seed if args.seed > 0 else rng.seed()
    print(f"Using seed: {seed}")

    z_random = get_noise(
        1, new_h, new_w, device=device, dtype=torch.bfloat16, seed=seed
    )

    if args.offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(device), clip.to(device)

    inp_source = prepare(t5, clip, init_latent, prompt=args.source_prompt)
    inp_target = prepare(t5, clip, init_latent, prompt=args.target_prompt)
    inp_random = prepare(t5, clip, z_random, prompt=args.target_prompt)
    key_word_index = get_word_index(args.source_prompt, args.edit_object, t5)

    info = {
        "feature_path": args.feature_path,
        "feature": {},
        "inject_step": args.inject,
        "sampling_solver": "fireflow",
        "latent_h": new_h // 16,
        "latent_w": new_w // 16,
        "key_word_index": key_word_index,
        "edit_type": args.edit_type,
        "kv_mix_ratio": args.kv_mix_ratio,
        "indices": [],
        "kv_mix": True,
        "kv_mask": True,
        "use_soft_mask": args.use_soft_mask,
        "soft_mask_gamma": args.soft_mask_gamma,
        "use_adaptive_kv": args.use_adaptive_kv,
        "use_channel_ls": args.use_channel_ls,
        "channel_ls_temp": args.channel_ls_temp,
        "inject_schedule": args.inject_schedule,
    }

    timesteps = get_schedule(
        args.num_steps,
        inp_source["img"].shape[1],
        shift=(args.name != "flux-schnell"),
    )
    info["mask_time"] = timesteps[-2]

    if args.offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(device)

    # --- Phase 1: inversion (original path) -------------------------
    print("Phase 1: Inverting source image (original AdaEdit path)...")
    z_inv, info = denoise_fireflow(
        model, **inp_source, timesteps=timesteps,
        guidance=1.0, inverse=True, info=info,
    )

    # --- Phase 2: channel-selective latents-shift (original) --------
    if args.edit_type != "style":
        print("Phase 2: Applying latent perturbation (original path)...")
        if args.use_channel_ls:
            z_inv = channel_selective_latents_shift(
                z_inv, inp_random["img"], info["indices"],
                alpha=args.ls_ratio, temperature=args.channel_ls_temp,
            )
        else:
            z_inv = latents_shift(
                z_inv, inp_random["img"], info["indices"], alpha=args.ls_ratio
            )

    inp_target["img"] = z_inv
    timesteps = get_schedule(
        args.num_steps,
        inp_target["img"].shape[1],
        shift=(args.name != "flux-schnell"),
    )

    # --- Phase 3: target pass ---------------------------------------
    controller = make_controller(
        args.mode,
        kp=args.kp, ki=args.ki, kd=args.kd,
        target_drift=args.target_drift,
        base_alpha=args.base_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        integral_clip=args.integral_clip,
    )

    drift_meter = DriftMeter(mode=args.drift_metric)

    if args.mode == "no_injection":
        # Ablation: run target pass with KV-Mix fully disabled. The
        # layers short-circuit to the no-injection attention path,
        # so the edit proceeds with zero source-side preservation.
        print("Phase 3: target pass — mode=no_injection (KV-Mix disabled)")
        prev_kv_mix = info["kv_mix"]
        info["kv_mix"] = False
        x_out, info = denoise_fireflow(
            model, **inp_target, timesteps=timesteps,
            guidance=args.guidance, inverse=False, info=info,
        )
        info["kv_mix"] = prev_kv_mix
        info.setdefault("adaptive_log", [])
    elif args.mode in ("original", "scheduled_fixed") or controller is None:
        print(f"Phase 3: target pass — mode={args.mode} (original sampler)")
        x_out, info = denoise_fireflow(
            model, **inp_target, timesteps=timesteps,
            guidance=args.guidance, inverse=False, info=info,
        )
        info.setdefault("adaptive_log", [])
    else:
        print(
            f"Phase 3: target pass — mode={args.mode} "
            f"(drift={args.drift_metric}, combine={args.combine})"
        )
        x_out, info = denoise_fireflow_adaptive(
            model, **inp_target, timesteps=timesteps,
            info=info, z_init=z_inv.clone(),
            controller=controller, drift_meter=drift_meter,
            combine=args.combine, guidance=args.guidance,
        )

    # --- decode & save ---------------------------------------------
    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x_out.device)

    batch_x = unpack(x_out.float(), new_h, new_w)

    run_name = args.run_name or f"{args.mode}_{args.inject_schedule}_inject{args.inject}"
    run_dir = _next_run_dir(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # pick the first (and only) sample
    x = batch_x[0].unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img_out = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    out_fn = os.path.join(run_dir, "edited.jpg")
    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = f"AdaEdit-Adaptive[{args.mode}]"
    exif_data[ExifTags.Base.Make] = "AdaEdit"
    exif_data[ExifTags.Base.Model] = args.name
    exif_data[ExifTags.Base.ImageDescription] = args.target_prompt
    img_out.save(out_fn, exif=exif_data, quality=95, subsampling=0)
    print(f"✓ Saved edited image: {out_fn}")

    # --- logs -------------------------------------------------------
    if args.log_controller:
        log_payload = {
            "mode": args.mode,
            "drift_metric": args.drift_metric,
            "combine": args.combine,
            "kp": args.kp, "ki": args.ki, "kd": args.kd,
            "target_drift": args.target_drift,
            "base_alpha": args.base_alpha,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "integral_clip": args.integral_clip,
            "kv_mix_ratio": args.kv_mix_ratio,
            "inject_schedule": args.inject_schedule,
            "inject_step": args.inject,
            "num_steps": args.num_steps,
            "seed": seed,
            "per_step": info.get("adaptive_log", []),
            "controller_history": (
                [asdict(s) for s in controller.history] if controller else []
            ),
        }
        log_fn = os.path.join(run_dir, "adaptive_log.json")
        with open(log_fn, "w") as f:
            json.dump(log_payload, f, indent=2)
        print(f"✓ Saved controller log: {log_fn}")

    # --- optional metrics -------------------------------------------
    if args.metrics:
        try:
            from src.flux.adaptive.metrics import compute_metrics

            pixel_mask = _indices_to_pixel_mask(
                info.get("indices", None),
                info["latent_h"], info["latent_w"],
                new_h, new_w,
            )
            metrics = compute_metrics(
                source=source_image,
                output=img_out,
                edit_mask=pixel_mask,
            )
            metrics_fn = os.path.join(run_dir, "metrics.json")
            with open(metrics_fn, "w") as f:
                json.dump(metrics, f, indent=2)
            print("Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"Warning: metrics computation failed: {e}")

    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f}s.")
    return run_dir


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "AdaEdit-Adaptive: experimental runner with PD/PID control "
            "of injection strength."
        )
    )

    # basic
    p.add_argument("--name", "-n", default="flux-dev", type=str)
    p.add_argument("--source_img", "-i", required=True, type=str)
    p.add_argument("--source_prompt", "-sp", required=True, type=str)
    p.add_argument("--target_prompt", "-tp", required=True, type=str)
    p.add_argument("--output_dir", "-o", default="outputs_adaptive", type=str)
    p.add_argument("--run_name", default=None, type=str)

    # edit
    p.add_argument("--edit_object", default="", type=str)
    p.add_argument(
        "--edit_type", default="change", type=str,
        choices=["add", "remove", "change", "style"],
    )

    # sampler
    p.add_argument("--num_steps", default=15, type=int)
    p.add_argument("--guidance", default=4.0, type=float)
    p.add_argument("--seed", default=0, type=int)

    # injection
    p.add_argument("--inject", default=4, type=int)
    p.add_argument(
        "--inject_schedule", default="sigmoid", type=str,
        choices=["binary", "sigmoid", "cosine", "linear"],
    )
    p.add_argument("--kv_mix_ratio", default=0.9, type=float)

    # latents-shift
    p.add_argument("--ls_ratio", default=0.25, type=float)
    p.add_argument("--use_channel_ls", action="store_true")
    p.add_argument("--channel_ls_temp", default=1.0, type=float)

    # optional AdaEdit bits
    p.add_argument("--use_soft_mask", action="store_true")
    p.add_argument("--soft_mask_gamma", default=5.0, type=float)
    p.add_argument("--use_adaptive_kv", action="store_true")

    # adaptive control
    p.add_argument("--mode", default="pd_adaptive", type=str,
                   choices=list(ADAPTIVE_MODES))
    p.add_argument(
        "--drift_metric", default="latent_init", type=str,
        choices=["latent_init", "latent_step", "latent_init_soft",
                 "latent_combined"],
    )
    p.add_argument(
        "--combine", default="multiply", type=str,
        choices=["multiply", "replace"],
        help="'multiply' -> delta = base * alpha * w(i); "
             "'replace' -> delta = base * alpha",
    )
    p.add_argument("--kp", default=1.0, type=float)
    p.add_argument("--ki", default=0.1, type=float)
    p.add_argument("--kd", default=0.2, type=float)
    p.add_argument("--target_drift", default=0.05, type=float)
    p.add_argument("--base_alpha", default=1.0, type=float)
    p.add_argument("--alpha_min", default=0.2, type=float)
    p.add_argument("--alpha_max", default=1.2, type=float)
    p.add_argument("--integral_clip", default=1.0, type=float)

    # logging / metrics
    p.add_argument("--log_controller", action="store_true", default=True,
                   help="Write per-step controller+drift log JSON (default on)")
    p.add_argument("--no_log_controller", dest="log_controller",
                   action="store_false")
    p.add_argument("--metrics", action="store_true",
                   help="Compute lightweight bg-preservation / edit metrics")

    # system
    p.add_argument("--feature_path", default="features", type=str)
    p.add_argument("--offload", action="store_true")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
