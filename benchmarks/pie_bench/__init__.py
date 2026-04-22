"""PIE-Bench integration.

The loader (pure numpy, no torch) is imported eagerly so scripts that
only parse/inspect the dataset don't pull torch in. The runner, which
calls our pipeline, is re-exported lazily via __getattr__.
"""

from .loader import (
    PIE_BENCH_TO_ADAEDIT,
    PIESample,
    iter_samples,
    load_sample,
    sample_pie,
)

__all__ = [
    "PIE_BENCH_TO_ADAEDIT",
    "PIESample",
    "iter_samples",
    "load_sample",
    "run_pie_samples",
    "sample_pie",
]


def __getattr__(name):
    if name == "run_pie_samples":
        from .runner import run_pie_samples
        return run_pie_samples
    raise AttributeError(f"module 'benchmarks.pie_bench' has no attribute {name!r}")
