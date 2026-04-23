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

from typing import List, Optional

import torch


_SUPPORTED = (
    "latent_init",
    "latent_step",
    "latent_init_soft",
    "latent_combined",
    # phase-10 signal variants
    "latent_relative",
    "latent_init_cosine",
    "latent_init_p90",
    "latent_relative_cosine",
)


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


def _masked_cosine_distance(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> float:
    """1 - cosine_similarity over the masked preservation subspace.

    Flattens [B, L, C] -> per-sample [L*C] vector with mask applied to
    each token, then takes cosine across the preservation subspace.
    Range ~[0, 2].
    """
    af = a.float()
    bf = b.float()
    m = mask  # [L, 1]
    af_m = af * m
    bf_m = bf * m
    a_flat = af_m.reshape(af_m.shape[0], -1)
    b_flat = bf_m.reshape(bf_m.shape[0], -1)
    num = (a_flat * b_flat).sum(dim=-1)
    den = a_flat.norm(dim=-1) * b_flat.norm(dim=-1) + 1e-8
    cos = (num / den).clamp(-1.0, 1.0)
    return float((1.0 - cos).mean().item())


def _masked_percentile_err(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, q: float = 0.9
) -> float:
    """qth-quantile of per-token squared error, restricted to mask>0 tokens."""
    diff = (a.float() - b.float()).pow(2).sum(dim=-1)  # [B, L]
    m = mask.squeeze(-1)  # [L]
    if m.dim() == 1:
        m = m.unsqueeze(0).expand_as(diff)
    sel = diff[m > 0]
    if sel.numel() == 0:
        return 0.0
    return float(torch.quantile(sel, q).item())


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
        step_idx: Optional[int] = None,
        source_traj: Optional[List[torch.Tensor]] = None,
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
        elif self.mode == "latent_relative":
            ref = self._lookup_ref(z_init, step_idx, source_traj)
            mask = _preservation_mask(
                img.shape[1], img.device, mask_dtype, indices, soft_mask
            )
            value = _masked_mse(img, ref, mask)
        elif self.mode == "latent_init_cosine":
            mask = _preservation_mask(
                img.shape[1], img.device, mask_dtype, indices, soft_mask
            )
            value = _masked_cosine_distance(img, z_init, mask)
        elif self.mode == "latent_init_p90":
            mask = _preservation_mask(
                img.shape[1], img.device, mask_dtype, indices, soft_mask
            )
            value = _masked_percentile_err(img, z_init, mask, q=0.9)
        elif self.mode == "latent_relative_cosine":
            ref = self._lookup_ref(z_init, step_idx, source_traj)
            mask = _preservation_mask(
                img.shape[1], img.device, mask_dtype, indices, soft_mask
            )
            rel = _masked_mse(img, ref, mask)
            cos = _masked_cosine_distance(img, z_init, mask)
            value = 0.5 * rel + 0.5 * cos
        else:
            raise RuntimeError(f"unreachable drift mode: {self.mode}")

        return value

    @staticmethod
    def _lookup_ref(
        z_init: torch.Tensor,
        step_idx: Optional[int],
        source_traj: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        if source_traj is None or step_idx is None or len(source_traj) == 0:
            return z_init
        i = max(0, min(int(step_idx), len(source_traj) - 1))
        return source_traj[i]
