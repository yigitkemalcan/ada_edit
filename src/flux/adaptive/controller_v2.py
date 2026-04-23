"""
Adaptive-injection controllers for v2 variants.

v1 (controller.py) has a single-objective PD/PID loop that only tracks
preservation drift. Two problems with that on PIE-Bench: it drives
kv_mix_ratio low to minimize preservation drift, which suppresses
editing; and it has no signal for "the edit isn't happening", so CLIP-T
and CLIP-dir suffer.

This module adds controllers that address those:

    - DualObjectiveController   variant A (dual_objective)
        Two error signals, one control law. Preservation error raises
        alpha (more injection -> more preservation); edit error lowers
        alpha (under-editing -> less injection -> more editing freedom).

    - ScheduledTargetController variant D (scheduled_target)
        PDController with a time-varying target_drift. Relaxed early,
        tight late (or reverse) — lets editing happen early and locks
        fidelity late.

Variants B, C, and E reuse the vanilla PDController from controller.py
but are driven differently by their samplers (see
sampling_adaptive_v2.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .controller import PDController, ControllerStep


# =============================================================================
# Variant A — DualObjectiveController
# =============================================================================

@dataclass
class DualControllerStep:
    step: int
    drift_pres: float
    drift_edit: float
    e_pres: float
    e_edit: float
    p_pres: float
    d_pres: float
    p_edit: float
    d_edit: float
    control: float
    alpha: float


class DualObjectiveController:
    """
    MIMO PD controller balancing preservation and edit drifts.

    Control law:
        e_pres(t) = drift_pres(t)  - target_pres
        e_edit(t) = target_edit    - drift_edit(t)    # inverted: want drift_edit >= target_edit
        control   = kp_p * e_pres + kd_p * d/dt(e_pres)
                  - kp_e * e_edit - kd_e * d/dt(e_edit)
        alpha     = clamp(base_alpha + control, alpha_min, alpha_max)

    The preservation branch follows the v1 PD contract; the edit branch
    has its sign flipped in the control law so that *under-editing*
    (drift_edit < target_edit, so e_edit > 0) drives alpha DOWN, which
    reduces source-side KV-Mix and gives target K/V more room.

    target_edit is derived from target_pres via edit_pres_ratio so the
    two setpoints stay scale-matched with a single knob:
        target_edit = edit_pres_ratio * target_pres

    If `clip_pres_below_zero` is True the preservation error is floored
    at 0 — i.e. the controller never rewards over-preservation. That
    avoids the two branches double-pushing alpha down when preservation
    is already below target.
    """

    def __init__(
        self,
        kp_p: float = 1.5,
        kd_p: float = 0.2,
        kp_e: float = 1.0,
        kd_e: float = 0.1,
        target_pres: float = 0.02,
        edit_pres_ratio: float = 3.0,
        base_alpha: float = 1.0,
        alpha_min: float = 0.2,
        alpha_max: float = 1.2,
        clip_pres_below_zero: bool = True,
    ) -> None:
        self.kp_p = kp_p
        self.kd_p = kd_p
        self.kp_e = kp_e
        self.kd_e = kd_e
        self.target_pres = target_pres
        self.target_edit = edit_pres_ratio * target_pres
        self.edit_pres_ratio = edit_pres_ratio
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.clip_pres_below_zero = clip_pres_below_zero
        self._prev_e_pres: Optional[float] = None
        self._prev_e_edit: Optional[float] = None
        self._t: int = 0
        self.history: List[DualControllerStep] = []

    def reset(self) -> None:
        self._prev_e_pres = None
        self._prev_e_edit = None
        self._t = 0
        self.history = []

    def step(self, drift_pres: float, drift_edit: float) -> float:
        e_pres = drift_pres - self.target_pres
        if self.clip_pres_below_zero and e_pres < 0.0:
            e_pres = 0.0
        e_edit = self.target_edit - drift_edit

        prev_p = self._prev_e_pres if self._prev_e_pres is not None else e_pres
        prev_e = self._prev_e_edit if self._prev_e_edit is not None else e_edit
        de_pres = e_pres - prev_p
        de_edit = e_edit - prev_e

        p_pres = self.kp_p * e_pres
        d_pres = self.kd_p * de_pres
        p_edit = self.kp_e * e_edit
        d_edit = self.kd_e * de_edit
        control = (p_pres + d_pres) - (p_edit + d_edit)

        alpha = self.base_alpha + control
        if alpha < self.alpha_min:
            alpha = self.alpha_min
        elif alpha > self.alpha_max:
            alpha = self.alpha_max

        self.history.append(
            DualControllerStep(
                step=self._t,
                drift_pres=drift_pres,
                drift_edit=drift_edit,
                e_pres=e_pres,
                e_edit=e_edit,
                p_pres=p_pres,
                d_pres=d_pres,
                p_edit=p_edit,
                d_edit=d_edit,
                control=control,
                alpha=alpha,
            )
        )
        self._prev_e_pres = e_pres
        self._prev_e_edit = e_edit
        self._t += 1
        return alpha


# =============================================================================
# Variant D — ScheduledTargetController
# =============================================================================


class ScheduledTargetController(PDController):
    """
    PDController with a time-varying target_drift.

    target_t = schedule(i / (num_steps - 1)) where schedule is one of:
        cosine_high_low: td_high * (1-s) + td_low * s   with s = 0.5*(1 - cos(pi*u))
        cosine_low_high: reverse of the above
        linear         : td_high * (1-u) + td_low * u (if high->low) or reverse

    Call .step(drift, step_idx) — the base PD .step() is single-arg, so
    this subclass adds `step_idx` as a required kwarg to update the
    target before delegating.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        td_high: float,
        td_low: float,
        td_profile: str = "cosine_high_low",
        kp: float = 1.0,
        kd: float = 0.2,
        base_alpha: float = 1.0,
        alpha_min: float = 0.2,
        alpha_max: float = 1.2,
    ) -> None:
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if td_profile not in ("cosine_high_low", "cosine_low_high", "linear"):
            raise ValueError(
                f"unknown td_profile {td_profile!r}. "
                "Expected cosine_high_low / cosine_low_high / linear."
            )
        super().__init__(
            kp=kp,
            kd=kd,
            target_drift=td_high,  # placeholder; real target set per step
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        self.num_steps = num_steps
        self.td_high = td_high
        self.td_low = td_low
        self.td_profile = td_profile

    def _compute_target(self, step_idx: int) -> float:
        if self.num_steps <= 1:
            return self.td_high
        u = step_idx / (self.num_steps - 1)
        u = max(0.0, min(1.0, u))
        if self.td_profile == "linear":
            return (1.0 - u) * self.td_high + u * self.td_low
        s = 0.5 * (1.0 - math.cos(math.pi * u))
        if self.td_profile == "cosine_high_low":
            return (1.0 - s) * self.td_high + s * self.td_low
        # cosine_low_high
        return (1.0 - s) * self.td_low + s * self.td_high

    def step(self, drift: float, step_idx: int = 0) -> float:  # type: ignore[override]
        self.target = self._compute_target(step_idx)
        return super().step(drift)


# =============================================================================
# Factory
# =============================================================================


def make_controller_v2(
    mode: str,
    *,
    num_steps: int,
    # shared
    base_alpha: float = 1.0,
    alpha_min: float = 0.2,
    alpha_max: float = 1.2,
    # v1-style PD gains (reused by B, C, E and the preservation branch of A/D)
    kp: float = 1.0,
    kd: float = 0.2,
    target_drift: float = 0.02,
    # dual-objective
    kp_p: float = 1.5,
    kd_p: float = 0.2,
    kp_e: float = 1.0,
    kd_e: float = 0.1,
    target_pres: float = 0.02,
    edit_pres_ratio: float = 3.0,
    clip_pres_below_zero: bool = True,
    # scheduled-target
    td_high: float = 0.08,
    td_low: float = 0.015,
    td_profile: str = "cosine_high_low",
):
    """
    Return the controller instance for `mode`. Variants B/C/E reuse the
    plain PDController from controller.py — their samplers handle the
    variant-specific logic (phase switching / asymmetric mixing / xattn
    release).
    """
    if mode == "dual_objective":
        return DualObjectiveController(
            kp_p=kp_p,
            kd_p=kd_p,
            kp_e=kp_e,
            kd_e=kd_e,
            target_pres=target_pres,
            edit_pres_ratio=edit_pres_ratio,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            clip_pres_below_zero=clip_pres_below_zero,
        )
    if mode == "scheduled_target":
        return ScheduledTargetController(
            num_steps=num_steps,
            td_high=td_high,
            td_low=td_low,
            td_profile=td_profile,
            kp=kp,
            kd=kd,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    if mode in ("two_phase_switch", "asymmetric_region", "xattn_boost"):
        return PDController(
            kp=kp,
            kd=kd,
            target_drift=target_drift,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    raise ValueError(
        f"make_controller_v2: unknown v2 mode {mode!r}."
    )


V2_MODES = (
    "dual_objective",
    "two_phase_switch",
    "asymmetric_region",
    "scheduled_target",
    "xattn_boost",
)
