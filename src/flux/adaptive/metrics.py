"""
Standard image-editing metrics: CLIP, LPIPS, SSIM, PSNR.

These are the four metrics reported in the AdaEdit paper. Each is
computed using the canonical community implementation:

  - CLIP  : transformers CLIPModel (openai/clip-vit-large-patch14).
            Returns both CLIP-T (edit vs target prompt) and CLIP-I
            (edit vs source image) cosine similarities.
  - LPIPS : Richard Zhang's `lpips` package, AlexNet backbone — the
            default used by nearly every image-editing paper.
  - SSIM  : scikit-image `structural_similarity`.
  - PSNR  : scikit-image `peak_signal_noise_ratio`.

When an ``edit_mask`` is supplied, background-only variants (prefixed
``*_bg``) are also computed. The mask marks the edit region with 1 and
the preservation region with 0. Background metrics quantify how well
the pipeline left the non-edited area alone:

  - psnr_bg  : PSNR computed over background pixels only.
  - ssim_bg  : SSIM map averaged over background pixels only.
  - lpips_bg : LPIPS between source and a composite where the edit
               region of the output is replaced by source pixels, so
               only background differences contribute.

CLIP is semantic / pooled, so a masked variant is not meaningful and
is intentionally omitted.

Models (CLIP + LPIPS) are loaded lazily and cached in module globals so
that repeated calls during a sweep do not re-download / re-instantiate.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image


_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_LPIPS_MODEL = None


def _to_np_uint8(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize_to(arr_uint8: np.ndarray, size_hw) -> np.ndarray:
    h, w = size_hw
    if arr_uint8.shape[:2] == (h, w):
        return arr_uint8
    pil = Image.fromarray(arr_uint8).resize((w, h), Image.BICUBIC)
    return np.array(pil)


def _get_clip(device: torch.device):
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-large-patch14"
        _CLIP_MODEL = CLIPModel.from_pretrained(name).to(device).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(name)
    elif next(_CLIP_MODEL.parameters()).device != device:
        _CLIP_MODEL = _CLIP_MODEL.to(device)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _get_lpips(device: torch.device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    elif next(_LPIPS_MODEL.parameters()).device != device:
        _LPIPS_MODEL = _LPIPS_MODEL.to(device)
    return _LPIPS_MODEL


def _as_tensor(x) -> torch.Tensor:
    """Extract a tensor from a HF ModelOutput or return x unchanged.

    transformers 4.x returned a raw tensor from
    CLIPModel.get_image_features / get_text_features. transformers 5.x
    wraps the result in a BaseModelOutputWithPooling. This helper makes
    the call site version-agnostic.
    """
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output",
                 "last_hidden_state"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(
        f"Cannot extract tensor from CLIP output of type {type(x).__name__}"
    )


def _prep_mask(
    mask: Optional[np.ndarray], h: int, w: int
) -> Optional[np.ndarray]:
    """Return a float32 HxW mask in [0,1] aligned to the source image,
    or None if no usable mask was provided."""
    if mask is None:
        return None
    m = np.asarray(mask).astype(np.float32)
    if m.ndim == 3:
        m = m.mean(axis=-1)
    if m.size == 0:
        return None
    if m.max() > 1.5:
        m = m / 255.0
    if m.shape != (h, w):
        pil = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((w, h), Image.NEAREST)
        m = np.array(pil).astype(np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)


@torch.inference_mode()
def compute_metrics(
    source: Union[Image.Image, np.ndarray],
    output: Union[Image.Image, np.ndarray],
    target_prompt: Optional[str] = None,
    source_prompt: Optional[str] = None,
    edit_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute the four standard image-editing metrics between a source
    image and an edited output.

    Returned keys (present when the underlying library succeeds):
        psnr / ssim / lpips     : whole-image preservation vs source
        psnr_bg / ssim_bg /
        lpips_bg                : same, restricted to background
                                  (preservation) region — only when
                                  a non-trivial edit_mask is provided
        clip_i                  : CLIP cosine sim, edit ↔ source image
        clip_t                  : CLIP cosine sim, edit ↔ target_prompt
        clip_dir                : CLIP directional sim
                                  (Δimage ↔ Δtext), only when both
                                  source_prompt and target_prompt are
                                  provided
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = _to_np_uint8(source)
    out = _to_np_uint8(output)
    out = _resize_to(out, src.shape[:2])
    h, w = src.shape[:2]

    mask_f = _prep_mask(edit_mask, h, w)
    # Background-only metrics are only meaningful if the mask carves
    # out a real foreground AND a real background.
    has_mask = (
        mask_f is not None
        and mask_f.sum() > 1e-3
        and (1.0 - mask_f).sum() > 1e-3
    )

    metrics: Dict[str, float] = {}

    # --- SSIM / PSNR (scikit-image) ---------------------------------
    try:
        from skimage.metrics import (
            peak_signal_noise_ratio,
            structural_similarity,
        )
        metrics["psnr"] = float(
            peak_signal_noise_ratio(src, out, data_range=255)
        )
        ssim_val, ssim_map = structural_similarity(
            src, out, channel_axis=-1, data_range=255, full=True,
        )
        metrics["ssim"] = float(ssim_val)

        if has_mask:
            bg = 1.0 - mask_f
            src_f = src.astype(np.float32)
            out_f = out.astype(np.float32)
            sq_err = (src_f - out_f) ** 2  # HxWx3
            num = float((sq_err * bg[..., None]).sum())
            den = float(bg.sum() * src_f.shape[-1] + 1e-8)
            bg_mse = num / den
            metrics["psnr_bg"] = (
                99.0 if bg_mse <= 0
                else float(10.0 * math.log10(255.0 ** 2 / max(bg_mse, 1e-10)))
            )
            # ssim_map is per-channel (HxWx3); average channels first
            ssim_map_m = (
                ssim_map.mean(axis=-1) if ssim_map.ndim == 3 else ssim_map
            )
            metrics["ssim_bg"] = float(
                (ssim_map_m * bg).sum() / (bg.sum() + 1e-8)
            )
    except Exception as e:
        print(f"[metrics] SSIM/PSNR failed: {e}")

    # --- LPIPS (AlexNet) --------------------------------------------
    try:
        model = _get_lpips(device)

        def _to_lpips_tensor(a: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)
            return (t / 127.5 - 1.0).to(device)

        dist = model(_to_lpips_tensor(src), _to_lpips_tensor(out))
        metrics["lpips"] = float(dist.squeeze().item())

        if has_mask:
            # Replace the foreground of `out` with source pixels so the
            # LPIPS deep features only see background differences.
            m3 = mask_f[..., None]
            composite = (
                out.astype(np.float32) * (1.0 - m3)
                + src.astype(np.float32) * m3
            )
            composite = np.clip(composite, 0, 255).astype(np.uint8)
            dist_bg = model(
                _to_lpips_tensor(src), _to_lpips_tensor(composite)
            )
            metrics["lpips_bg"] = float(dist_bg.squeeze().item())
    except Exception as e:
        print(f"[metrics] LPIPS failed: {e}")

    # --- CLIP --------------------------------------------------------
    try:
        clip_model, clip_proc = _get_clip(device)
        src_pil = Image.fromarray(src)
        out_pil = Image.fromarray(out)

        img_inputs = clip_proc(
            images=[src_pil, out_pil], return_tensors="pt"
        ).to(device)
        img_feats = _as_tensor(clip_model.get_image_features(**img_inputs))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        src_feat, out_feat = img_feats[0], img_feats[1]

        metrics["clip_i"] = float((src_feat * out_feat).sum().item())

        if target_prompt is not None:
            txt_inputs = clip_proc(
                text=[target_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            tgt_feat = _as_tensor(clip_model.get_text_features(**txt_inputs))
            tgt_feat = tgt_feat / tgt_feat.norm(dim=-1, keepdim=True)
            metrics["clip_t"] = float((out_feat * tgt_feat[0]).sum().item())

            if source_prompt is not None:
                src_txt = clip_proc(
                    text=[source_prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                src_txt_feat = _as_tensor(
                    clip_model.get_text_features(**src_txt)
                )
                src_txt_feat = src_txt_feat / src_txt_feat.norm(
                    dim=-1, keepdim=True
                )
                d_img = out_feat - src_feat
                d_txt = tgt_feat[0] - src_txt_feat[0]
                d_img = d_img / (d_img.norm() + 1e-8)
                d_txt = d_txt / (d_txt.norm() + 1e-8)
                metrics["clip_dir"] = float((d_img * d_txt).sum().item())
    except Exception as e:
        print(f"[metrics] CLIP failed: {e}")

    return metrics
