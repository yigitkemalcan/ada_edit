"""
Light-weight optional metrics for comparing edited vs source images.

Designed to run without extra dependencies (numpy + PIL only). All
metrics are *optional*: the adaptive runner can skip this module
entirely when --metrics is not passed.

Metrics provided:
  - masked_psnr_bg:  PSNR on the background (preservation) region.
  - masked_mse_bg:   mean squared error on the background region.
  - ssim_bg:         simplified SSIM on the background region.
  - mean_abs_edit:   mean abs difference inside the edit region
                     (proxy for whether editing actually happened).

The mask is expected as a pixel-space bool/float array of the source
image's resolution. If no mask is provided, metrics fall back to
whole-image values (still useful for rough comparison).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from PIL import Image


def _to_float01(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    else:
        arr = img
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    return arr


def _align(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    if src.shape == out.shape:
        return out
    h, w = src.shape[:2]
    pil = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((w, h), resample=Image.BICUBIC)
    return np.array(pil).astype(np.float32) / 255.0


def _prep_mask(
    mask: Optional[np.ndarray],
    h: int,
    w: int,
) -> np.ndarray:
    if mask is None:
        return np.ones((h, w), dtype=np.float32)
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m.mean(axis=-1)
    if m.max() > 1.5:
        m = m / 255.0
    if m.shape != (h, w):
        pil = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((w, h), resample=Image.NEAREST)
        m = np.array(pil).astype(np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)


def _masked_mse(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    diff = (a - b) ** 2
    if diff.ndim == 3:
        w3 = weight[..., None]
    else:
        w3 = weight
    num = float((diff * w3).sum())
    den = float(w3.sum() * (diff.shape[-1] if diff.ndim == 3 else 1) + 1e-8)
    return num / den


def _simple_ssim(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    """Global single-scale SSIM restricted to weighted region.
    Not a full sliding-window SSIM -- sufficient for coarse comparison.
    """
    a = a.mean(axis=-1) if a.ndim == 3 else a
    b = b.mean(axis=-1) if b.ndim == 3 else b
    w = weight
    s = float(w.sum() + 1e-8)
    mu_a = float((a * w).sum() / s)
    mu_b = float((b * w).sum() / s)
    va = float(((a - mu_a) ** 2 * w).sum() / s)
    vb = float(((b - mu_b) ** 2 * w).sum() / s)
    cov = float(((a - mu_a) * (b - mu_b) * w).sum() / s)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return num / (den + 1e-8)


def compute_metrics(
    source: Image.Image | np.ndarray,
    output: Image.Image | np.ndarray,
    edit_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Args:
        source:     source image (before edit).
        output:     edited image. If a different resolution, will be
                    resampled to source resolution.
        edit_mask:  optional pixel-space mask, 1.0 inside edit region,
                    0.0 in preservation region. Any grayscale array
                    accepted; is resized with NEAREST to source size.
    """
    src = _to_float01(source)
    out = _to_float01(output)
    out = _align(src, out)
    h, w = src.shape[:2]
    m = _prep_mask(edit_mask, h, w)
    bg = 1.0 - m  # preservation weight

    bg_mse = _masked_mse(src, out, bg)
    if bg_mse <= 0:
        bg_psnr = 99.0
    else:
        bg_psnr = 10.0 * math.log10(1.0 / max(bg_mse, 1e-10))

    bg_ssim = _simple_ssim(src, out, bg)

    if m.sum() > 1e-6:
        edit_abs = float(
            (np.abs(src - out).mean(axis=-1) * m).sum() / (m.sum() + 1e-8)
        )
    else:
        edit_abs = 0.0

    return {
        "masked_mse_bg": bg_mse,
        "masked_psnr_bg": bg_psnr,
        "ssim_bg": bg_ssim,
        "mean_abs_edit": edit_abs,
    }
