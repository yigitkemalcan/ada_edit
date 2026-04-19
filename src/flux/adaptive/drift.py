"""
Drift signals for adaptive injection control.

A drift value quantifies how much the edited trajectory has deviated
from the source/reference in regions that should be preserved. It is
computed once per denoising step on the token sequence the model is
operating on (pre-unpack latents, shape [B, L, C]).

The metric is pluggable: a DriftMeter instance holds configuration +
light state (previous latent for step-based metrics) and exposes a
single update(img, ctx) -> float method.
"""

from __future__ import annotations

from typing import Optional

import torch


_SUPPORTED = ("latent_init", "latent_step", "latent_init_soft", "latent_combined")


def _preservation_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    indices: Optional[torch.Tensor],
    soft_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Build a per-token weight in [0, 1] that is 1 in preservation
    regions and 0 in edit regions.
    Shape: [L, 1].
    """
    if soft_mask is not None and soft_mask.numel() == seq_len:
        m = soft_mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        weight = 1.0 - m
    elif indices is not None and torch.is_tensor(indices) and indices.numel() > 0:
        weight = torch.ones(seq_len, device=device, dtype=torch.float32)
        idx = indices.to(device=device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < seq_len)]
        if idx.numel() > 0:
            weight[idx] = 0.0
    else:
        weight = torch.ones(seq_len, device=device, dtype=torch.float32)
    return weight.to(dtype=dtype).unsqueeze(-1)


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2)
    num = (diff * mask).sum()
    den = mask.sum() * a.shape[-1] + 1e-8
    return float((num / den).item())


def compute_drift(
    img: torch.Tensor,
    reference: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    soft_mask: Optional[torch.Tensor] = None,
) -> float:
    """Stateless helper: masked MSE between two token tensors."""
    mask = _preservation_mask(
        img.shape[1], img.device, torch.float32, indices, soft_mask
    )
    return _masked_mse(img, reference, mask)


class DriftMeter:
    """
    Keeps optional previous-latent state and returns a scalar drift
    value at each denoising step.

    Modes:
        - latent_init:      MSE(img, z_init) on preservation tokens.
        - latent_step:      MSE(img, img_{t-1}) on preservation tokens.
                            Proxy for drift velocity.
        - latent_init_soft: latent_init but uses soft mask weight
                            (1 - soft_mask) if one is available.
        - latent_combined:  0.5 * latent_init + 0.5 * latent_step.

    Call reset() between runs; the module guarantees no leakage.
    """

    def __init__(self, mode: str = "latent_init"):
        if mode not in _SUPPORTED:
            raise ValueError(
                f"Unsupported drift mode {mode!r}. Choose from {_SUPPORTED}."
            )
        self.mode = mode
        self._prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._prev = None

    def update(
        self,
        img: torch.Tensor,
        z_init: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        soft_mask: Optional[torch.Tensor] = None,
    ) -> float:
        mask_dtype = torch.float32
        if self.mode in ("latent_init", "latent_init_soft"):
            sm = soft_mask if self.mode == "latent_init_soft" else None
            idx = indices if self.mode == "latent_init" else None
            mask = _preservation_mask(img.shape[1], img.device, mask_dtype, idx, sm)
            value = _masked_mse(img, z_init, mask)
        elif self.mode == "latent_step":
            if self._prev is None:
                value = 0.0
            else:
                mask = _preservation_mask(
                    img.shape[1], img.device, mask_dtype, indices, soft_mask
                )
                value = _masked_mse(img, self._prev, mask)
            self._prev = img.detach().clone()
        elif self.mode == "latent_combined":
            mask = _preservation_mask(
                img.shape[1], img.device, mask_dtype, indices, soft_mask
            )
            init_v = _masked_mse(img, z_init, mask)
            step_v = 0.0 if self._prev is None else _masked_mse(img, self._prev, mask)
            self._prev = img.detach().clone()
            value = 0.5 * init_v + 0.5 * step_v
        else:
            raise RuntimeError(f"unreachable drift mode: {self.mode}")

        return value
