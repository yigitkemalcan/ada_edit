"""
Adaptive closed-loop control extensions for AdaEdit.

This package is a separate experimental path. The original AdaEdit
pipeline (adaedit.py + src/flux/sampling.py::denoise_fireflow) is
untouched by anything in here.
"""

from .controller import PDController, PIDController, make_controller
from .drift import DriftMeter, compute_drift
from .sampling_adaptive import denoise_fireflow_adaptive

__all__ = [
    "PDController",
    "PIDController",
    "make_controller",
    "DriftMeter",
    "compute_drift",
    "denoise_fireflow_adaptive",
]
