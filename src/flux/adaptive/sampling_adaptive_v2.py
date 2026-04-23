"""
v2 adaptive FireFlow samplers — one per new variant.

Each sampler mirrors the structure of denoise_fireflow_adaptive (v1 in
sampling_adaptive.py): second-order FireFlow inner step, drift meter
measured per step, controller mutates info['kv_mix_ratio'] before the
model is called. They differ only in how alpha_t and the effective
kv_mix ratio are computed:

    variant A  dual_objective    — uses DualObjectiveController plus a
                                    second meter (EditProgressMeter).
    variant B  two_phase_switch  — deterministic alpha/kv override for
                                    the first edit_fraction*T steps,
                                    then PD controller active.
    variant C  asymmetric_region — writes info['kv_mix_ratio_edit'] and
                                    info['kv_mix_ratio_preserve']; the
                                    guarded branch in layers.py picks
                                    them up. Requires use_soft_mask.
    variant D  scheduled_target  — ScheduledTargetController (time-
                                    varying target_drift). Otherwise
                                    identical to v1 path.
    variant E  xattn_boost       — reads info['xattn_edit_score']
                                    captured in layers.py and applies a
                                    release factor when under-attending
                                    to the edit word.

Shared helpers:
    - _fireflow_inner_step: the 2-call FireFlow iteration (identical to
      v1 sampler lines 127-168).
    - _effective_kv_ratio:  imported from v1 sampler.

Nothing in v1 is edited — this file is additive.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor

from ..model import Flux
from ..sampling import get_progressive_inject_schedule
from .controller import PDController
from .controller_v2 import (
    DualObjectiveController,
    ScheduledTargetController,
)
from .drift import DriftMeter
from .drift_v2 import EditProgressMeter
from .sampling_adaptive import _effective_kv_ratio


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _fireflow_inner_step(
    model,
    img: Tensor,
    t_curr: float,
    t_prev: float,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    guidance_vec: Tensor,
    info: dict,
    next_step_velocity: Optional[Tensor],
):
    """FireFlow 2nd-order step — same as denoise_fireflow target pass."""
    t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
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
    return img, info, next_step_velocity


def _prep_step_common(info, num_steps: int):
    schedule_type = info.get("inject_schedule", "binary")
    inject_weights = get_progressive_inject_schedule(
        num_steps, info["inject_step"], schedule_type
    )
    w_floor = float(info.get("inject_weight_floor", 0.0))
    if w_floor > 0.0:
        inject_weights = [max(float(w), w_floor) for w in inject_weights]
    base_kv = float(info["kv_mix_ratio"])
    indices = info.get("indices", None)
    if not torch.is_tensor(indices):
        indices = None
    soft_mask = info.get("soft_mask", None)
    source_traj = info.get("source_trajectory", None)
    return inject_weights, base_kv, indices, soft_mask, source_traj


# -----------------------------------------------------------------------------
# Variant D — scheduled_target
# -----------------------------------------------------------------------------


@torch.inference_mode()
def denoise_fireflow_scheduled(
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
    controller: ScheduledTargetController,
    drift_meter: DriftMeter,
    combine: str = "multiply",
    guidance: float = 4.0,
):
    num_steps = len(timesteps[:-1])
    inject_weights, base_kv, indices, soft_mask, source_traj = _prep_step_common(info, num_steps)
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    controller.reset()
    drift_meter.reset()
    adaptive_log = []
    next_step_velocity = None

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        drift_val = drift_meter.update(
            img=img, z_init=z_init, indices=indices, soft_mask=soft_mask,
            step_idx=i, source_traj=source_traj,
        )
        alpha_t = controller.step(drift_val, step_idx=i)
        w_i = float(inject_weights[i])
        info["kv_mix_ratio"] = _effective_kv_ratio(base_kv, alpha_t, w_i, combine)
        info["inject"] = w_i > 0.05
        info["inject_weight"] = w_i

        img, info, next_step_velocity = _fireflow_inner_step(
            model, img, t_curr, t_prev, img_ids, txt, txt_ids, vec,
            guidance_vec, info, next_step_velocity,
        )
        adaptive_log.append({
            "step": i,
            "t_curr": float(t_curr),
            "t_prev": float(t_prev),
            "inject_weight": w_i,
            "drift": drift_val,
            "target_drift_t": controller.target,
            "alpha": alpha_t,
            "kv_mix_ratio": info["kv_mix_ratio"],
            "base_kv_mix_ratio": base_kv,
            "variant": "scheduled_target",
        })

    info["kv_mix_ratio"] = base_kv
    info["adaptive_log"] = adaptive_log
    return img, info


# -----------------------------------------------------------------------------
# Variant B — two_phase_switch
# -----------------------------------------------------------------------------


@torch.inference_mode()
def denoise_fireflow_two_phase(
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
    controller: PDController,
    drift_meter: DriftMeter,
    combine: str = "multiply",
    guidance: float = 4.0,
    edit_fraction: float = 0.4,
    alpha_edit: float = 0.3,
    kv_mix_edit: float = 0.6,
    kv_mix_preserve: float = 0.9,
    phase_ramp_steps: int = 2,
):
    """
    Two-phase schedule. Edit phase (first edit_fraction*T steps):
        - force alpha = alpha_edit
        - override base kv with kv_mix_edit
    Preserve phase: activate PDController with base kv = kv_mix_preserve.
    A linear ramp between phases (length phase_ramp_steps) smooths the
    transition; controller _prev_error is cleared at the boundary.
    """
    num_steps = len(timesteps[:-1])
    inject_weights, _, indices, soft_mask, source_traj = _prep_step_common(info, num_steps)
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    controller.reset()
    drift_meter.reset()
    adaptive_log = []
    next_step_velocity = None

    boundary = int(round(edit_fraction * num_steps))
    boundary = max(1, min(num_steps, boundary))
    ramp = max(0, min(phase_ramp_steps, num_steps - boundary))

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        drift_val = drift_meter.update(
            img=img, z_init=z_init, indices=indices, soft_mask=soft_mask,
            step_idx=i, source_traj=source_traj,
        )

        if i < boundary:
            phase = "edit"
            alpha_t = alpha_edit
            base_kv_t = kv_mix_edit
        elif i < boundary + ramp:
            # Linear ramp from edit to preserve configuration.
            u = (i - boundary + 1) / (ramp + 1)
            phase = "ramp"
            if i == boundary:
                controller._prev_error = None  # reset derivative memory
            alpha_pd = controller.step(drift_val)
            alpha_t = (1.0 - u) * alpha_edit + u * alpha_pd
            base_kv_t = (1.0 - u) * kv_mix_edit + u * kv_mix_preserve
        else:
            phase = "preserve"
            if i == boundary and ramp == 0:
                controller._prev_error = None
            alpha_t = controller.step(drift_val)
            base_kv_t = kv_mix_preserve

        w_i = float(inject_weights[i])
        info["kv_mix_ratio"] = _effective_kv_ratio(base_kv_t, alpha_t, w_i, combine)
        info["inject"] = w_i > 0.05
        info["inject_weight"] = w_i

        img, info, next_step_velocity = _fireflow_inner_step(
            model, img, t_curr, t_prev, img_ids, txt, txt_ids, vec,
            guidance_vec, info, next_step_velocity,
        )
        adaptive_log.append({
            "step": i,
            "t_curr": float(t_curr),
            "t_prev": float(t_prev),
            "inject_weight": w_i,
            "drift": drift_val,
            "alpha": alpha_t,
            "kv_mix_ratio": info["kv_mix_ratio"],
            "base_kv_t": base_kv_t,
            "phase": phase,
            "variant": "two_phase_switch",
        })

    # Restore base_kv from the original info
    info["kv_mix_ratio"] = kv_mix_preserve
    info["adaptive_log"] = adaptive_log
    return img, info


# -----------------------------------------------------------------------------
# Variant A — dual_objective
# -----------------------------------------------------------------------------


@torch.inference_mode()
def denoise_fireflow_dual(
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
    controller: DualObjectiveController,
    drift_meter: DriftMeter,
    edit_meter: EditProgressMeter,
    combine: str = "multiply",
    guidance: float = 4.0,
):
    num_steps = len(timesteps[:-1])
    inject_weights, base_kv, indices, soft_mask, source_traj = _prep_step_common(info, num_steps)
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    controller.reset()
    drift_meter.reset()
    edit_meter.reset()
    adaptive_log = []
    next_step_velocity = None

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        d_pres = drift_meter.update(
            img=img, z_init=z_init, indices=indices, soft_mask=soft_mask,
            step_idx=i, source_traj=source_traj,
        )
        d_edit = edit_meter.update(
            img=img,
            z_init=z_init,
            indices=indices,
            soft_mask=soft_mask,
            preservation_drift=d_pres,
        )
        alpha_t = controller.step(d_pres, d_edit)

        w_i = float(inject_weights[i])
        info["kv_mix_ratio"] = _effective_kv_ratio(base_kv, alpha_t, w_i, combine)
        info["inject"] = w_i > 0.05
        info["inject_weight"] = w_i

        img, info, next_step_velocity = _fireflow_inner_step(
            model, img, t_curr, t_prev, img_ids, txt, txt_ids, vec,
            guidance_vec, info, next_step_velocity,
        )
        adaptive_log.append({
            "step": i,
            "t_curr": float(t_curr),
            "t_prev": float(t_prev),
            "inject_weight": w_i,
            "drift_pres": d_pres,
            "drift_edit": d_edit,
            "alpha": alpha_t,
            "kv_mix_ratio": info["kv_mix_ratio"],
            "base_kv_mix_ratio": base_kv,
            "target_pres": controller.target_pres,
            "target_edit": controller.target_edit,
            "variant": "dual_objective",
        })

    info["kv_mix_ratio"] = base_kv
    info["adaptive_log"] = adaptive_log
    return img, info


# -----------------------------------------------------------------------------
# Variant C — asymmetric_region
# -----------------------------------------------------------------------------


@torch.inference_mode()
def denoise_fireflow_asymmetric(
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
    controller: PDController,
    drift_meter: DriftMeter,
    combine: str = "multiply",
    guidance: float = 4.0,
    kv_mix_edit: float = 0.45,
    kv_mix_preserve: float = 0.9,
):
    """
    Region-conditional kv_mix. The sampler writes info['kv_mix_ratio_edit']
    and info['kv_mix_ratio_preserve']; the guarded branch in layers.py
    (added by the matching diff) builds a per-token ratio tensor from
    soft_mask. Requires info['use_soft_mask'] to be True.
    """
    if not info.get("use_soft_mask", False):
        raise RuntimeError(
            "asymmetric_region requires use_soft_mask=True so a per-token "
            "mask exists to blend between kv_mix_edit and kv_mix_preserve."
        )

    num_steps = len(timesteps[:-1])
    inject_weights, _, indices, soft_mask, source_traj = _prep_step_common(info, num_steps)
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    controller.reset()
    drift_meter.reset()
    adaptive_log = []
    next_step_velocity = None

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        d_pres = drift_meter.update(
            img=img, z_init=z_init, indices=indices, soft_mask=soft_mask,
            step_idx=i, source_traj=source_traj,
        )
        alpha_t = controller.step(d_pres)
        w_i = float(inject_weights[i])

        r_edit_eff = _effective_kv_ratio(kv_mix_edit, alpha_t, w_i, combine)
        r_pres_eff = _effective_kv_ratio(kv_mix_preserve, alpha_t, w_i, combine)

        # v1 path still reads info['kv_mix_ratio']; set to the preservation
        # value so any unguarded fall-through stays safe.
        info["kv_mix_ratio"] = r_pres_eff
        info["kv_mix_ratio_edit"] = r_edit_eff
        info["kv_mix_ratio_preserve"] = r_pres_eff
        info["inject"] = w_i > 0.05
        info["inject_weight"] = w_i

        img, info, next_step_velocity = _fireflow_inner_step(
            model, img, t_curr, t_prev, img_ids, txt, txt_ids, vec,
            guidance_vec, info, next_step_velocity,
        )
        adaptive_log.append({
            "step": i,
            "t_curr": float(t_curr),
            "t_prev": float(t_prev),
            "inject_weight": w_i,
            "drift": d_pres,
            "alpha": alpha_t,
            "kv_mix_edit": r_edit_eff,
            "kv_mix_preserve": r_pres_eff,
            "variant": "asymmetric_region",
        })

    # Clean up — remove per-region keys so subsequent v1 runs re-see a
    # scalar ratio.
    info.pop("kv_mix_ratio_edit", None)
    info.pop("kv_mix_ratio_preserve", None)
    info["kv_mix_ratio"] = kv_mix_preserve
    info["adaptive_log"] = adaptive_log
    return img, info


# -----------------------------------------------------------------------------
# Variant E — xattn_boost
# -----------------------------------------------------------------------------


@torch.inference_mode()
def denoise_fireflow_xattn(
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
    controller: PDController,
    drift_meter: DriftMeter,
    combine: str = "multiply",
    guidance: float = 4.0,
    target_xattn: float = 0.08,
    xattn_release_threshold: float = 0.02,
    release_factor: float = 0.7,
):
    """
    Cross-attention boost. The guarded hook in layers.py writes
    info['xattn_edit_score'] once per step. If the model is
    under-attending to the edit word AND preservation drift is below
    target (so we have budget to loosen), scale alpha down by
    release_factor for that step.
    """
    num_steps = len(timesteps[:-1])
    inject_weights, base_kv, indices, soft_mask, source_traj = _prep_step_common(info, num_steps)
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    controller.reset()
    drift_meter.reset()
    adaptive_log = []
    next_step_velocity = None

    # Turn on the capture hook in layers.py for this pass.
    info["capture_xattn_to_editword"] = True
    info.pop("xattn_edit_score", None)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        d_pres = drift_meter.update(
            img=img, z_init=z_init, indices=indices, soft_mask=soft_mask,
            step_idx=i, source_traj=source_traj,
        )
        alpha_t = controller.step(d_pres)

        # Previous step's captured xattn (or None on the first step).
        xattn_prev = info.get("xattn_edit_score", None)
        released = False
        if xattn_prev is not None:
            e_xattn = target_xattn - float(xattn_prev)
            e_pres = d_pres - controller.target
            if e_xattn > xattn_release_threshold and e_pres < 0.0:
                alpha_t = alpha_t * release_factor
                released = True

        w_i = float(inject_weights[i])
        info["kv_mix_ratio"] = _effective_kv_ratio(base_kv, alpha_t, w_i, combine)
        info["inject"] = w_i > 0.05
        info["inject_weight"] = w_i

        img, info, next_step_velocity = _fireflow_inner_step(
            model, img, t_curr, t_prev, img_ids, txt, txt_ids, vec,
            guidance_vec, info, next_step_velocity,
        )
        adaptive_log.append({
            "step": i,
            "t_curr": float(t_curr),
            "t_prev": float(t_prev),
            "inject_weight": w_i,
            "drift": d_pres,
            "alpha": alpha_t,
            "kv_mix_ratio": info["kv_mix_ratio"],
            "base_kv_mix_ratio": base_kv,
            "xattn_edit_score": info.get("xattn_edit_score", None),
            "released": released,
            "variant": "xattn_boost",
        })

    info["capture_xattn_to_editword"] = False
    info["kv_mix_ratio"] = base_kv
    info["adaptive_log"] = adaptive_log
    return img, info


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------


def dispatch_v2_sampler(mode: str):
    """Return the sampler callable for mode."""
    if mode == "dual_objective":
        return denoise_fireflow_dual
    if mode == "two_phase_switch":
        return denoise_fireflow_two_phase
    if mode == "asymmetric_region":
        return denoise_fireflow_asymmetric
    if mode == "scheduled_target":
        return denoise_fireflow_scheduled
    if mode == "xattn_boost":
        return denoise_fireflow_xattn
    raise ValueError(f"dispatch_v2_sampler: unknown v2 mode {mode!r}")
