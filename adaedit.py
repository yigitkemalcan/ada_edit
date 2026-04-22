"""
AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing
Main entry point for image editing.
"""
import os
import re
import json
import time
import argparse
from glob import iglob
from dataclasses import dataclass

import torch
import numpy as np
from einops import rearrange
from PIL import Image, ExifTags
from transformers import pipeline

from src.flux.sampling import (
    denoise_fireflow,
    get_schedule,
    prepare,
    unpack,
    get_noise,
    latents_shift,
    channel_selective_latents_shift,
)
from src.flux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    get_word_index,
)

NSFW_THRESHOLD = 0.85


@dataclass
class EditConfig:
    """Configuration for image editing."""
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


@torch.inference_mode()
def encode_image(image_array, device, ae):
    """Encode image to latent space."""
    img = torch.from_numpy(image_array).permute(2, 0, 1).float() / 127.5 - 1
    img = img.unsqueeze(0).to(device)
    latent = ae.encode(img).to(torch.bfloat16)
    return latent


@torch.inference_mode()
def main(args):
    """Main editing function."""
    torch.set_grad_enabled(False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load NSFW classifier (optional)
    try:
        nsfw_classifier = pipeline("image-classification",
                                  model="Falconsai/nsfw_image_detection",
                                  device=device)
    except Exception as e:
        print(f"Warning: NSFW classifier not available: {e}")
        nsfw_classifier = None

    # Validate model name
    if args.name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model: {args.name}, choose from {available}")

    # Load models
    print("Loading models...")
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(args.name, device="cpu" if args.offload else device)
    ae = load_ae(args.name, device="cpu" if args.offload else device)

    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(device)

    # Load and preprocess source image
    print(f"Loading source image: {args.source_img}")
    source_image = np.array(Image.open(args.source_img).convert('RGB'))
    h, w = source_image.shape[:2]

    # Ensure dimensions are multiples of 16
    new_h = h if h % 16 == 0 else h - h % 16
    new_w = w if w % 16 == 0 else w - w % 16
    source_image = source_image[:new_h, :new_w, :]

    print(f"Image size: {new_w}x{new_h}")

    # Encode source image
    t0 = time.perf_counter()
    init_latent = encode_image(source_image, device, ae)

    # Setup random seed
    rng = torch.Generator(device="cpu")
    seed = args.seed if args.seed > 0 else rng.seed()
    print(f"Using seed: {seed}")

    # Prepare random noise for Latents-Shift
    z_random = get_noise(1, new_h, new_w, device=device,
                        dtype=torch.bfloat16, seed=seed)

    # Offload autoencoder, load text encoders
    if args.offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(device), clip.to(device)

    # Prepare text embeddings
    inp_source = prepare(t5, clip, init_latent, prompt=args.source_prompt)
    inp_target = prepare(t5, clip, init_latent, prompt=args.target_prompt)
    inp_random = prepare(t5, clip, z_random, prompt=args.target_prompt)

    # Get word index for mask extraction
    key_word_index = get_word_index(args.source_prompt, args.edit_object, t5)

    # Setup info dict for editing
    info = {
        'feature_path': args.feature_path,
        'feature': {},
        'inject_step': args.inject,
        'sampling_solver': 'fireflow',
        'latent_h': new_h // 16,
        'latent_w': new_w // 16,
        'key_word_index': key_word_index,
        'edit_type': args.edit_type,
        'kv_mix_ratio': args.kv_mix_ratio,
        'indices': [],
        'kv_mix': True,
        'kv_mask': True,
        # AdaEdit specific
        'use_soft_mask': args.use_soft_mask,
        'soft_mask_gamma': args.soft_mask_gamma,
        'use_adaptive_kv': args.use_adaptive_kv,
        'use_channel_ls': args.use_channel_ls,
        'channel_ls_temp': args.channel_ls_temp,
        'inject_schedule': args.inject_schedule,
    }

    # Get timesteps
    timesteps = get_schedule(args.num_steps, inp_source["img"].shape[1],
                            shift=(args.name != "flux-schnell"))
    info['mask_time'] = timesteps[-2]

    # Offload text encoders, load model
    if args.offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(device)

    # Phase 1: Inversion with feature caching
    print("Phase 1: Inverting source image...")
    z_inv, info = denoise_fireflow(model, **inp_source, timesteps=timesteps,
                                   guidance=1.0, inverse=True, info=info)

    # Phase 2: Channel-Selective Latent Perturbation
    if args.edit_type != 'style':
        print("Phase 2: Applying latent perturbation...")
        if args.use_channel_ls:
            z_inv = channel_selective_latents_shift(
                z_inv, inp_random["img"], info['indices'],
                alpha=args.ls_ratio, temperature=args.channel_ls_temp
            )
        else:
            z_inv = latents_shift(z_inv, inp_random["img"],
                                 info['indices'], alpha=args.ls_ratio)

    inp_target["img"] = z_inv
    timesteps = get_schedule(args.num_steps, inp_target["img"].shape[1],
                            shift=(args.name != "flux-schnell"))

    # Phase 3: Sampling with Progressive Injection
    print("Phase 3: Generating edited image...")
    x_out, _ = denoise_fireflow(model, **inp_target, timesteps=timesteps,
                                guidance=args.guidance, inverse=False, info=info)

    # Decode to pixel space
    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x_out.device)

    batch_x = unpack(x_out.float(), new_h, new_w)

    # Save output
    for x in batch_x:
        x = x.unsqueeze(0)

        # Generate output filename
        output_name = os.path.join(args.output_dir,
                                  f"{args.inject_schedule}_inject{args.inject}_img_{{idx}}.jpg")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*"))
                  if re.search(r"img_[0-9]+\.jpg$", fn)]
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1 if fns else 0

        # Decode latent
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving to {fn}")

        # Convert to PIL image
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        # NSFW check
        if nsfw_classifier is not None:
            nsfw_score = [x["score"] for x in nsfw_classifier(img)
                         if x["label"] == "nsfw"][0]
        else:
            nsfw_score = 0.0

        if nsfw_score < NSFW_THRESHOLD:
            # Save with metadata
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AdaEdit"
            exif_data[ExifTags.Base.Make] = "AdaEdit"
            exif_data[ExifTags.Base.Model] = args.name
            exif_data[ExifTags.Base.ImageDescription] = args.target_prompt
            img.save(fn, exif=exif_data, quality=95, subsampling=0)
            print(f"✓ Saved: {fn}")

            if args.metrics:
                try:
                    from src.flux.adaptive.metrics import compute_metrics
                    from adaedit_adaptive import _indices_to_pixel_mask
                    pixel_mask = _indices_to_pixel_mask(
                        info.get("indices", None),
                        info["latent_h"], info["latent_w"],
                        new_h, new_w,
                    )
                    metrics = compute_metrics(
                        source=source_image,
                        output=img,
                        target_prompt=args.target_prompt,
                        source_prompt=args.source_prompt,
                        edit_mask=pixel_mask,
                        device=device,
                    )
                    metrics_fn = fn.replace(".jpg", ".metrics.json")
                    with open(metrics_fn, "w") as f:
                        json.dump(metrics, f, indent=2)
                    print("Metrics:")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")
                    print(f"✓ Saved metrics: {metrics_fn}")
                except Exception as e:
                    print(f"Warning: metrics computation failed: {e}")
        else:
            print("✗ Image may contain NSFW content, not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing'
    )

    # Basic arguments
    parser.add_argument('--name', '-n', default='flux-dev', type=str,
                       help='Model name (default: flux-dev)')
    parser.add_argument('--source_img', '-i', required=True, type=str,
                       help='Path to source image')
    parser.add_argument('--source_prompt', '-sp', required=True, type=str,
                       help='Description of source image')
    parser.add_argument('--target_prompt', '-tp', required=True, type=str,
                       help='Target editing prompt')
    parser.add_argument('--output_dir', '-o', default='outputs', type=str,
                       help='Output directory (default: outputs)')

    # Editing parameters
    parser.add_argument('--edit_object', default='', type=str,
                       help='Object to edit (for mask extraction)')
    parser.add_argument('--edit_type', default='change', type=str,
                       choices=['add', 'remove', 'change', 'style'],
                       help='Type of edit (default: change)')

    # Sampling parameters
    parser.add_argument('--num_steps', default=15, type=int,
                       help='Number of sampling steps (default: 15)')
    parser.add_argument('--guidance', default=4.0, type=float,
                       help='Guidance scale (default: 4.0)')
    parser.add_argument('--seed', default=0, type=int,
                       help='Random seed (0 for random, default: 0)')

    # Injection parameters
    parser.add_argument('--inject', default=4, type=int,
                       help='Injection step threshold (default: 4)')
    parser.add_argument('--inject_schedule', default='sigmoid', type=str,
                       choices=['binary', 'sigmoid', 'cosine', 'linear'],
                       help='Injection schedule type (default: sigmoid)')
    parser.add_argument('--kv_mix_ratio', default=0.9, type=float,
                       help='KV-Mix ratio (default: 0.9)')

    # Latent perturbation parameters
    parser.add_argument('--ls_ratio', default=0.25, type=float,
                       help='Latents-Shift strength (default: 0.25)')
    parser.add_argument('--use_channel_ls', action='store_true',
                       help='Use channel-selective Latents-Shift')
    parser.add_argument('--channel_ls_temp', default=1.0, type=float,
                       help='Temperature for channel importance (default: 1.0)')

    # Optional modules
    parser.add_argument('--use_soft_mask', action='store_true',
                       help='Use soft mask instead of binary mask')
    parser.add_argument('--soft_mask_gamma', default=5.0, type=float,
                       help='Soft mask sharpness (default: 5.0)')
    parser.add_argument('--use_adaptive_kv', action='store_true',
                       help='Use layer-adaptive KV-Mix ratio')

    # System parameters
    parser.add_argument('--feature_path', default='features', type=str,
                       help='Path to save cached features (default: features)')
    parser.add_argument('--offload', action='store_true',
                       help='Enable model offloading for low VRAM')

    # Metrics
    parser.add_argument('--metrics', action='store_true', default=True,
                       help='Compute CLIP / LPIPS / SSIM / PSNR (default on)')
    parser.add_argument('--no_metrics', dest='metrics', action='store_false',
                       help='Disable metric computation')

    args = parser.parse_args()
    main(args)
