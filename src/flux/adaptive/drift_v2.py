"""
Edit-region progress meter — mirror of DriftMeter but measuring drift
INSIDE the edit mask rather than the preservation region.

Motivation. The v1 controller uses only preservation drift (DriftMeter
in drift.py), so the only error signal is "did we preserve enough?".
That drives kv_mix_ratio down and the controller cannot tell when the
edit is starved. EditProgressMeter produces the complementary signal:
how far the current latent has moved in the edit tokens. Together the
two meters let a MIMO controller (DualObjectiveController in
controller_v2.py) trade off preservation against edit strength.

Operation.
    edit mask        = 1 - preservation_mask   (pointwise)
    drift_edit_init  = MSE(img, z_init) over edit tokens
    drift_edit_step  = MSE(img, img_{t-1}) over edit tokens
    drift_edit_soft  = drift_edit_init but driven by soft_mask
    drift_edit_norm  = drift_edit_init / (preservation_drift + eps)

`edit_normalized` requires the caller to pass the preservation drift
from the same step (already computed by DriftMeter); it collapses the
two scalars into a single dimensionless signal which is robust to
per-image scale variation.

State is reset per run via .reset() (same contract as DriftMeter).
"""

from __future__ import annotations

from typing import Optional

import torch

from .drift import _preservation_mask, _masked_mse


_SUPPORTED_EDIT = (
    "edit_init",
    "edit_step",
    "edit_init_soft",
    "edit_normalized",
)


def _edit_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    indices: Optional[torch.Tensor],
    soft_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Build per-token weight in [0, 1] that is 1 in edit tokens and 0
    elsewhere. Complement of _preservation_mask.
    Shape: [L, 1].
    """
    pres = _preservation_mask(seq_len, device, dtype, indices, soft_mask)
    return (1.0 - pres).clamp(0.0, 1.0)


class EditProgressMeter:
    """
    Measures latent movement in the edit region per step.

    Higher value = more editing. Controller should push alpha down
    (less injection) when this is low.

    Modes:
        edit_init       : MSE(img, z_init) on edit tokens.
        edit_step       : MSE(img, img_{t-1}) on edit tokens (velocity).
        edit_init_soft  : edit_init but uses soft_mask as the weight.
        edit_normalized : edit_init / (preservation_drift + eps). Use
                          when the two meters are run in lockstep; the
                          caller passes the preservation scalar in.

    When `indices` is empty and `soft_mask` is absent the meter has no
    edit region to measure and returns 0.0 (the dual controller will
    then not act on edit error).
    """

    def __init__(self, mode: str = "edit_init"):
        if mode not in _SUPPORTED_EDIT:
            raise ValueError(
                f"Unsupported edit-drift mode {mode!r}. "
                f"Choose from {_SUPPORTED_EDIT}."
            )
        self.mode = mode
        self._prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._prev = None

    def _has_edit_region(
        self,
        seq_len: int,
        indices: Optional[torch.Tensor],
        soft_mask: Optional[torch.Tensor],
    ) -> bool:
        if soft_mask is not None and soft_mask.numel() == seq_len:
            return bool((soft_mask > 1e-4).any().item())
        if indices is not None and torch.is_tensor(indices) and indices.numel() > 0:
            return True
        return False

    def update(
        self,
        img: torch.Tensor,
        z_init: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        soft_mask: Optional[torch.Tensor] = None,
        preservation_drift: Optional[float] = None,
    ) -> float:
        seq_len = img.shape[1]
        if not self._has_edit_region(seq_len, indices, soft_mask):
            # No mask yet (very first inversion steps) — return 0.
            if self.mode in ("edit_step", "edit_normalized"):
                self._prev = img.detach().clone()
            return 0.0

        mask_dtype = torch.float32

        if self.mode == "edit_init":
            mask = _edit_mask(seq_len, img.device, mask_dtype, indices, None)
            return _masked_mse(img, z_init, mask)

        if self.mode == "edit_init_soft":
            mask = _edit_mask(seq_len, img.device, mask_dtype, None, soft_mask)
            return _masked_mse(img, z_init, mask)

        if self.mode == "edit_step":
            mask = _edit_mask(seq_len, img.device, mask_dtype, indices, soft_mask)
            if self._prev is None:
                value = 0.0
            else:
                value = _masked_mse(img, self._prev, mask)
            self._prev = img.detach().clone()
            return value

        if self.mode == "edit_normalized":
            mask = _edit_mask(seq_len, img.device, mask_dtype, indices, soft_mask)
            num = _masked_mse(img, z_init, mask)
            den = (preservation_drift if preservation_drift is not None else 0.0)
            return float(num / (den + 1e-6))

        raise RuntimeError(f"unreachable edit-drift mode: {self.mode}")
