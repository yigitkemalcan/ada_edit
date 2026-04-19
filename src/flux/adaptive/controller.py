"""
PD / PID controllers for adaptive injection strength.

Given drift_t at each timestep, produce alpha_t that modulates the
injection strength used by AdaEdit's KV-Mix. Higher alpha -> more
source injection -> more preservation. Lower alpha -> less injection
-> more editing freedom.

    error_t      = drift_t - target_drift
    derivative_t = error_t - error_{t-1}
    control_t    = kp * error_t + ki * I_t + kd * derivative_t
    alpha_t      = clamp(base_alpha + control_t, alpha_min, alpha_max)

State must be reset between runs; controllers track the sign/magnitude
of prior errors and leak across runs otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ControllerStep:
    step: int
    drift: float
    error: float
    proportional: float
    integral: float
    derivative: float
    control: float
    alpha: float


class PDController:
    def __init__(
        self,
        kp: float = 1.0,
        kd: float = 0.2,
        target_drift: float = 0.05,
        base_alpha: float = 1.0,
        alpha_min: float = 0.2,
        alpha_max: float = 1.2,
    ) -> None:
        self.kp = kp
        self.kd = kd
        self.target = target_drift
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._prev_error: Optional[float] = None
        self._t: int = 0
        self.history: List[ControllerStep] = []

    def reset(self) -> None:
        self._prev_error = None
        self._t = 0
        self.history = []

    def _integral_term(self, error: float) -> float:
        return 0.0

    def step(self, drift: float) -> float:
        error = drift - self.target
        prev = self._prev_error if self._prev_error is not None else error
        derivative = error - prev
        integral = self._integral_term(error)

        p_term = self.kp * error
        i_term = integral
        d_term = self.kd * derivative
        control = p_term + i_term + d_term

        alpha = self.base_alpha + control
        if alpha < self.alpha_min:
            alpha = self.alpha_min
        elif alpha > self.alpha_max:
            alpha = self.alpha_max

        self.history.append(
            ControllerStep(
                step=self._t,
                drift=drift,
                error=error,
                proportional=p_term,
                integral=i_term,
                derivative=d_term,
                control=control,
                alpha=alpha,
            )
        )
        self._prev_error = error
        self._t += 1
        return alpha


class PIDController(PDController):
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.2,
        target_drift: float = 0.05,
        base_alpha: float = 1.0,
        alpha_min: float = 0.2,
        alpha_max: float = 1.2,
        integral_clip: float = 1.0,
    ) -> None:
        super().__init__(
            kp=kp,
            kd=kd,
            target_drift=target_drift,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        self.ki = ki
        self.integral_clip = integral_clip
        self._integral: float = 0.0

    def reset(self) -> None:
        super().reset()
        self._integral = 0.0

    def _integral_term(self, error: float) -> float:
        self._integral += error
        # symmetric clip to keep integral from running away
        if self._integral > self.integral_clip:
            self._integral = self.integral_clip
        elif self._integral < -self.integral_clip:
            self._integral = -self.integral_clip
        return self.ki * self._integral


def make_controller(
    mode: str,
    *,
    kp: float = 1.0,
    ki: float = 0.1,
    kd: float = 0.2,
    target_drift: float = 0.05,
    base_alpha: float = 1.0,
    alpha_min: float = 0.2,
    alpha_max: float = 1.2,
    integral_clip: float = 1.0,
) -> Optional[PDController]:
    """
    Factory: returns None for modes that do not need a controller.
        - original          -> None (pure AdaEdit)
        - scheduled_fixed   -> None (alias for original)
        - fixed_soft        -> PDController with kp=kd=0 (constant base_alpha)
        - pd_adaptive       -> PDController
        - pid_adaptive      -> PIDController
    """
    if mode in ("original", "scheduled_fixed"):
        return None
    if mode == "fixed_soft":
        return PDController(
            kp=0.0,
            kd=0.0,
            target_drift=target_drift,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    if mode == "pd_adaptive":
        return PDController(
            kp=kp,
            kd=kd,
            target_drift=target_drift,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    if mode == "pid_adaptive":
        return PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            target_drift=target_drift,
            base_alpha=base_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            integral_clip=integral_clip,
        )
    raise ValueError(
        f"Unknown adaptive mode: {mode!r}. "
        "Expected one of: original, scheduled_fixed, fixed_soft, "
        "pd_adaptive, pid_adaptive."
    )
