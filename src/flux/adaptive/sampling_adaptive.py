"""
Adaptive FireFlow sampler for the target (editing) pass.

This mirrors src.flux.sampling.denoise_fireflow for inverse=False only,
but at each step it:

  1. Measures drift between the current latent and the inversion start
     latent z_init, restricted to preservation (non-edit) tokens.
  2. Steps the controller to get alpha_t.
  3. Multiplies AdaEdit's base injection strength (kv_mix_ratio) by
     alpha_t, optionally on top of the original progressive schedule.

The inversion pass must still use the original denoise_fireflow; only
the editing pass is replaced. This keeps feature caching and mask
extraction 100% identical to vanilla AdaEdit.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from ..model import Flux
from ..sampling import get_progressive_inject_schedule
from .controller import PDController
from .drift import DriftMeter


def _effective_kv_ratio(
    base: float,
    alpha: float,
    w_i: float,
    combine: str,
) -> float:
    if combine == "multiply":
        v = base * alpha * w_i
    elif combine == "replace":
        v = base * alpha
    else:
        raise ValueError(
            f"combine must be 'multiply' or 'replace', got {combine!r}"
        )
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@torch.inference_mode()
def denoise_fireflow_adaptive(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: List[float],
    info: dict,
    *,
    z_init: Tensor,
    controller: Optional[PDController],
    drift_meter: DriftMeter,
    combine: str = "multiply",
    guidance: float = 4.0,
):
    """
    Controller-driven target-pass sampler.

    Args:
        z_init:       The latent handed to this pass (i.e. the inversion
                      result, possibly after Latents-Shift). Serves as
                      the preservation reference.
        controller:   PDController/PIDController or None. When None, the
                      sampler behaves exactly like denoise_fireflow
                      (useful for sanity checks / scheduled_fixed mode).
        drift_meter:  Drift signal source.
        combine:      'multiply' -> delta = base * alpha * w(i)
                      'replace'  -> delta = base * alpha
    """
    num_steps = len(timesteps[:-1])
    schedule_type = info.get("inject_schedule", "binary")
    inject_weights = get_progressive_inject_schedule(
        num_steps, info["inject_step"], schedule_type
    )

    base_kv = float(info["kv_mix_ratio"])

    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    if controller is not None:
        controller.reset()
    drift_meter.reset()

    adaptive_log = []
    indices = info.get("indices", None)
    soft_mask = info.get("soft_mask", None)
    source_traj = info.get("source_trajectory", None)
    if not torch.is_tensor(indices):
        indices = None

    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # --- controller update ---------------------------------------
        drift_val = drift_meter.update(
            img=img,
            z_init=z_init,
            indices=indices,
            soft_mask=soft_mask,
            step_idx=i,
            source_traj=source_traj,
        )
        if controller is not None:
            alpha_t = controller.step(drift_val)
        else:
            alpha_t = 1.0

        w_i = float(inject_weights[i])
        info["kv_mix_ratio"] = _effective_kv_ratio(base_kv, alpha_t, w_i, combine)

        # gating threshold stays identical to the original sampler so
        # we do not change when injection is "on" vs "off"
        info["inject"] = w_i > 0.05
        info["inject_weight"] = w_i

        # --- model call (identical structure to denoise_fireflow) ----
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )
        info["t"] = t_curr
        info["inverse"] = False
        info["second_order"] = False

        if next_step_velocity is None:
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info,
            )
        else:
            pred = next_step_velocity

        img_mid = img + (t_prev - t_curr) / 2 * pred
        t_vec_mid = torch.full(
            (img.shape[0],),
            t_curr + (t_prev - t_curr) / 2,
            dtype=img.dtype,
            device=img.device,
        )
        info["second_order"] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
        )
        next_step_velocity = pred_mid
        img = img + (t_prev - t_curr) * pred_mid

        adaptive_log.append(
            {
                "step": i,
                "t_curr": float(t_curr),
                "t_prev": float(t_prev),
                "inject_weight": w_i,
                "drift": drift_val,
                "alpha": alpha_t,
                "kv_mix_ratio": info["kv_mix_ratio"],
                "base_kv_mix_ratio": base_kv,
            }
        )

    # restore base before returning so the caller's info dict is clean
    info["kv_mix_ratio"] = base_kv
    info["adaptive_log"] = adaptive_log
    return img, info
