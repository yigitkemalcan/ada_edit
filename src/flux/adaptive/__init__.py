"""
Adaptive closed-loop control extensions for AdaEdit.

This package is a separate experimental path. The original AdaEdit
pipeline (adaedit.py + src/flux/sampling.py::denoise_fireflow) is
untouched by anything in here.

v1 (controller.py, drift.py, sampling_adaptive.py) implements the
single-objective PD/PID loop.

v2 (controller_v2.py, drift_v2.py, sampling_adaptive_v2.py) adds five
new modes: dual_objective, two_phase_switch, asymmetric_region,
scheduled_target, xattn_boost.
"""

from .controller import PDController, PIDController, make_controller
from .controller_v2 import (
    DualObjectiveController,
    ScheduledTargetController,
    V2_MODES,
    make_controller_v2,
)
from .drift import DriftMeter, compute_drift
from .drift_v2 import EditProgressMeter
from .sampling_adaptive import denoise_fireflow_adaptive
from .sampling_adaptive_v2 import (
    denoise_fireflow_asymmetric,
    denoise_fireflow_dual,
    denoise_fireflow_scheduled,
    denoise_fireflow_two_phase,
    denoise_fireflow_xattn,
    dispatch_v2_sampler,
)

__all__ = [
    "PDController",
    "PIDController",
    "make_controller",
    "DriftMeter",
    "compute_drift",
    "denoise_fireflow_adaptive",
    # v2
    "DualObjectiveController",
    "ScheduledTargetController",
    "V2_MODES",
    "make_controller_v2",
    "EditProgressMeter",
    "denoise_fireflow_asymmetric",
    "denoise_fireflow_dual",
    "denoise_fireflow_scheduled",
    "denoise_fireflow_two_phase",
    "denoise_fireflow_xattn",
    "dispatch_v2_sampler",
]
