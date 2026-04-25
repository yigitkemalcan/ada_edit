"""
Cross-config PIE-Bench sweep.

Validates the phase-1/2 single-image findings on the real benchmark.
Every config is run on the identical 30-sample slice (``seed=0``, cat 8
excluded) used in phase 3, so mean metrics are directly comparable
across configs — no sampling noise between rows.

Usage (Colab):

    from sweep_pie_bench import run_pie_sweep

    results = run_pie_sweep(
        t5=t5, clip=clip, model=model, ae=ae,
        output_dir="outputs_pie_sweep",
        # configs=[...]   # override the default 13-config grid
    )

Per-config artefacts land under
``<output_dir>/<name>/<sample_key>/edited.jpg + metrics.json``, plus a
per-config ``pie_samples.csv``. An aggregate
``<output_dir>/sweep_pie_results.csv`` holds one row per config with
mean metrics and the config knobs — that's the file to load for
cross-config analysis.

Grid composition (13 configs, ~6 min each at 12.4 s/sample × 30 ≈
~80 min total):

  Vanilla AdaEdit (2)
    - baseline                  original, no extensions
    - paper_adaedit             original, 3 extensions on (paper default)

  PD without extensions (2)
    - pd_balanced               phase-2 balanced winner (kp=1.5, kd=0.2)
    - pd_kp0.5                  phase-2 clip_dir winner  (kp=0.5)

  PD with extensions (9)
    - pd_balanced_full          phase-2 balanced + 3 ext  (current best)
    - pd_kp0.5_full             PD + 3 ext, kp=0.5
    - pd_kp1.0_full             PD + 3 ext, kp=1.0 (phase-1 winner setting)
    - pd_ls0.25_full            PD + 3 ext, ls_ratio=0.25 (paper α)
    - pd_ls0.55_full            PD + 3 ext, ls_ratio=0.55
    - pd_ls0.65_full            PD + 3 ext, ls_ratio=0.65
    - pd_kv0.75_full            PD + 3 ext, kv_mix_ratio=0.75
    - pd_td0.05_full            PD + 3 ext, target_drift=0.05
    - pd_kd0.1_full             PD + 3 ext, kd=0.1

Questions this sweep answers:
  1. Do phase-2's kp/kd/ls/kv/td choices still win once extensions are
     enabled, or does the optimum shift?
  2. Is the ``kp=0.5 vs 1.5`` tie broken on multi-image data?
  3. Is the phase-2 `ls_ratio=0.45 > 0.25` pattern still true with ext?
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Optional

from benchmarks.pie_bench import run_pie_samples
from sweep import _METRIC_ORDER, _HIGHER_IS_BETTER, _fmt_cell


# Phase-2 balanced winner — shared base for every PD config below.
_PD_BASE: Dict[str, Any] = dict(
    mode="pd_adaptive",
    kv_mix_ratio=0.9,
    target_drift=0.02,
    ls_ratio=0.45,
    kp=1.5,
    kd=0.2,
)

# All three AdaEdit extensions on — the paper's full config.
_EXT_ON: Dict[str, Any] = dict(
    use_channel_ls=True,
    use_soft_mask=True,
    use_adaptive_kv=True,
)


def _default_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    # --- Vanilla AdaEdit (2) ---
    configs.append({
        "name": "baseline",
        "mode": "original",
        "kv_mix_ratio": 0.9,
        "ls_ratio": 0.25,
    })
    configs.append({
        "name": "paper_adaedit",
        "mode": "original",
        "kv_mix_ratio": 0.9,
        "ls_ratio": 0.25,
        **_EXT_ON,
    })

    # --- PD without extensions (2) ---
    configs.append({
        "name": "pd_balanced",
        **_PD_BASE,
    })
    configs.append({
        "name": "pd_kp0.5",
        **_PD_BASE, "kp": 0.5,
    })

    # --- PD with extensions (9) ---
    configs.append({
        "name": "pd_balanced_full",
        **_PD_BASE,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_kp0.5_full",
        **_PD_BASE, "kp": 0.5,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_kp1.0_full",
        **_PD_BASE, "kp": 1.0,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_ls0.25_full",
        **_PD_BASE, "ls_ratio": 0.25,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_ls0.55_full",
        **_PD_BASE, "ls_ratio": 0.55,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_ls0.65_full",
        **_PD_BASE, "ls_ratio": 0.65,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_kv0.75_full",
        **_PD_BASE, "kv_mix_ratio": 0.75,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_td0.05_full",
        **_PD_BASE, "target_drift": 0.05,
        **_EXT_ON,
    })
    configs.append({
        "name": "pd_kd0.1_full",
        **_PD_BASE, "kd": 0.1,
        **_EXT_ON,
    })

    return configs


# ---------------------------------------------------------------------
# v2 sweep: all 5 new variants + anchors (paper, pd_best)
# ---------------------------------------------------------------------

# Current best from phase-7/8: pd_adaptive, kv=0.30, kp=1.5, kd=0.2,
# target_drift=0.02, soft_mask ON, channel_ls + adaptive_kv OFF,
# ls_ratio=0.45. This is our "pd_best" anchor — the bar to beat.
_PD_BEST: Dict[str, Any] = dict(
    mode="pd_adaptive",
    kv_mix_ratio=0.30,
    target_drift=0.02,
    ls_ratio=0.45,
    kp=1.5,
    kd=0.2,
    use_soft_mask=True,
)


def v2_configs() -> List[Dict[str, Any]]:
    """
    Lean v2 PIE-Bench sweep: 7 configs — 2 anchors plus one
    representative per variant family. Picks the middle-of-the-grid
    defaults so the sweep is a fair "best-shot" comparison; once a
    family is identified as promising here, a follow-up sweep can
    widen the grid around that variant.

    Configs (7):
      - paper_adaedit  : original method with all 3 AdaEdit extensions
      - pd_best        : current v1 winner (single-objective PD)
      - dual_r3        : variant A (edit_pres_ratio=3)
      - phase_f04      : variant B (edit_fraction=0.4)
      - asym_45_90     : variant C (kv_mix_edit=0.45 / preserve=0.9)
      - sched_hl       : variant D (cosine_high_low profile)
      - xattn_07       : variant E (release_factor=0.7)
    """
    # Shared base for v2 rows: the winning pd_best knobs with mode
    # overridden per variant.
    def _pd(**over):
        out = dict(_PD_BEST)
        out.update(over)
        return out

    return [
        # --- anchors (2) ----
        {"name": "paper_adaedit",
         "mode": "original",
         "kv_mix_ratio": 0.9,
         "ls_ratio": 0.25,
         **_EXT_ON},
        {"name": "pd_best", **_PD_BEST},

        # --- one representative per variant family (5) ----
        {"name": "dual_r3",
         **_pd(mode="dual_objective"),
         "edit_pres_ratio": 3.0},
        {"name": "phase_f04",
         **_pd(mode="two_phase_switch"),
         "edit_fraction": 0.4,
         "kv_mix_edit": 0.45, "kv_mix_preserve": 0.9},
        {"name": "asym_45_90",
         **_pd(mode="asymmetric_region"),
         "kv_mix_edit": 0.45, "kv_mix_preserve": 0.9},
        {"name": "sched_hl",
         **_pd(mode="scheduled_target"),
         "td_high": 0.08, "td_low": 0.015,
         "td_profile": "cosine_high_low"},
        {"name": "xattn_07",
         **_pd(mode="xattn_boost"),
         "release_factor": 0.7},
    ]


def phase_f04_sweep() -> List[Dict[str, Any]]:
    """
    Marginal sweep around the phase-9 winner (phase_f04 / two_phase_switch).

    Anchor: edit_fraction=0.40, kv_mix_edit=0.45, kv_mix_preserve=0.90,
            alpha_edit=0.30 (all other knobs from _PD_BEST).

    One axis is varied at a time while the others stay at the anchor.
    The pd_best config is included as an external baseline.

    Configs (11):
      pd_best              — phase-7/8 single-objective anchor
      pf_base              — two_phase_switch at phase-9 winner values
      pf_ef30 … pf_ef50    — edit_fraction ∈ {0.30, 0.35, 0.45, 0.50}
      pf_kve30, pf_kve60   — kv_mix_edit ∈ {0.30, 0.60}
      pf_kvp75             — kv_mix_preserve = 0.75
      pf_ae15, pf_ae45     — alpha_edit ∈ {0.15, 0.45}

    Questions this sweep answers:
      1. What is the optimal edit_fraction for two_phase_switch on PIE-Bench?
      2. Does a lower kv_mix_edit (more source KV during edit phase) or
         higher (less source KV) improve fidelity / edit quality?
      3. Does relaxing kv_mix_preserve (0.75 vs 0.90) hurt preservation?
      4. Does a weaker or stronger alpha_edit gate change the balance?
    """
    # Anchor values matching the phase-9 phase_f04 config.
    _BASE_EF   = 0.40
    _BASE_KVE  = 0.45
    _BASE_KVP  = 0.90
    _BASE_AE   = 0.30

    def _pf(name, **over):
        cfg = dict(_PD_BEST)
        cfg.update(
            mode="two_phase_switch",
            edit_fraction=_BASE_EF,
            kv_mix_edit=_BASE_KVE,
            kv_mix_preserve=_BASE_KVP,
            alpha_edit=_BASE_AE,
        )
        cfg.update(over)
        cfg["name"] = name
        return cfg

    return [
        # External baseline
        {"name": "pd_best", **_PD_BEST},

        # Anchor (reproduces phase_f04 from phase 9)
        _pf("pf_base"),

        # edit_fraction axis
        _pf("pf_ef30", edit_fraction=0.30),
        _pf("pf_ef35", edit_fraction=0.35),
        _pf("pf_ef45", edit_fraction=0.45),
        _pf("pf_ef50", edit_fraction=0.50),

        # kv_mix_edit axis
        _pf("pf_kve30", kv_mix_edit=0.30),
        _pf("pf_kve60", kv_mix_edit=0.60),

        # kv_mix_preserve axis
        _pf("pf_kvp75", kv_mix_preserve=0.75),

        # alpha_edit axis
        _pf("pf_ae15", alpha_edit=0.15),
        _pf("pf_ae45", alpha_edit=0.45),
    ]


# ---------------------------------------------------------------------
# Phase-10 sweep: drift-signal variants under the pd_best controller
# ---------------------------------------------------------------------


def drift_signal_configs(
    td_relative: float = 0.02,
    td_cosine: float = 0.02,
    td_p90: float = 0.02,
    td_rel_cos: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Phase-10 sweep: compare drift signals under a fixed controller.

    All non-anchor rows use the phase-7/8 winner `_PD_BEST` (pd_adaptive,
    kv=0.30, kp=1.5, kd=0.2, soft_mask on) and differ only in
    `drift_metric` and its calibrated `target_drift`. Holding the
    controller fixed isolates signal effects, mirroring how phase 9
    held the signal fixed and varied the controller.

    Calibration protocol (run once, plug values in here):
      1. For each new drift_metric, run pd_best with kp=0, kd=0 on ~5
         PIE-Bench samples and record the median drift over the
         injection window.
      2. Set target_drift = 0.8 * median.

    Pass calibrated values via the keyword args; defaults match the
    latent_init scale (0.02) as a placeholder only.

    Rows:
      - paper_adaedit   : original method, kv=0.9
      - pd_best         : phase-7/8 anchor (latent_init, td=0.02)
      - drift_relative  : latent_relative (trajectory-matched MSE)
      - drift_cosine    : latent_init_cosine (directional)
      - drift_p90       : latent_init_p90   (worst-tokens)
      - drift_rel_cos   : latent_relative_cosine (only if V1 wins;
                          pass td_rel_cos to include)
    """
    def _pd(**over):
        out = dict(_PD_BEST)
        out.update(over)
        return out

    rows: List[Dict[str, Any]] = [
        {"name": "paper_adaedit",
         "mode": "original",
         "kv_mix_ratio": 0.9,
         "ls_ratio": 0.25,
         **_EXT_ON},
        {"name": "pd_best", **_PD_BEST},
        {"name": "drift_relative",
         **_pd(drift_metric="latent_relative", target_drift=td_relative)},
        {"name": "drift_cosine",
         **_pd(drift_metric="latent_init_cosine", target_drift=td_cosine)},
        {"name": "drift_p90",
         **_pd(drift_metric="latent_init_p90", target_drift=td_p90)},
    ]
    if td_rel_cos is not None:
        rows.append({
            "name": "drift_rel_cos",
            **_pd(drift_metric="latent_relative_cosine",
                  target_drift=td_rel_cos),
        })
    return rows


# ---------------------------------------------------------------------
# Phase-12 sweep: schedule × controller (combine mode + sigmoid floor)
# ---------------------------------------------------------------------


# Anchor for the phase_f04 controller (= phase-9 winner config).
_PHASE_F04: Dict[str, Any] = dict(
    _PD_BEST,
    mode="two_phase_switch",
    edit_fraction=0.40,
    kv_mix_edit=0.45,
    kv_mix_preserve=0.90,
    alpha_edit=0.30,
)


def phase12_configs(floor: float = 0.25) -> List[Dict[str, Any]]:
    """
    Phase-12 sweep: schedule × controller. Holds the drift signal at the
    Phase-7/8 default (latent_init) and varies how the per-step inject
    weight w(i) interacts with the controller's alpha. Two controller
    families × three combine modes:

    Combine modes
      - current   : kv_eff = base_kv * alpha * sigmoid_w(i)
                    (combine='multiply', floor=0.0 — same as pd_best)
      - full_adp  : kv_eff = base_kv * alpha
                    (combine='replace' — pure adaptive, no schedule)
      - sched_floor : kv_eff = base_kv * alpha * max(sigmoid_w(i), 0.25)
                    (combine='multiply', floor=0.25 — keeps the
                    controller active where sigmoid has decayed)

    Controllers
      - pd_best   : single-objective PD, kv=0.30 (Phase-7/8 winner)
      - phase_f04 : two_phase_switch, edit_fraction=0.40,
                    kv_mix_edit=0.45, kv_mix_preserve=0.90,
                    alpha_edit=0.30 (Phase-9 winner)

    Total: 6 rows.
    """
    def _ctrl(base, mode_label):
        if mode_label == "current":
            over = dict(combine="multiply", inject_weight_floor=0.0)
        elif mode_label == "full_adaptive":
            over = dict(combine="replace", inject_weight_floor=0.0)
        elif mode_label == "sched_floor":
            over = dict(combine="multiply", inject_weight_floor=floor)
        else:
            raise ValueError(mode_label)
        return {**base, **over}

    rows: List[Dict[str, Any]] = []
    for ctrl_name, ctrl_base in (("pd_best", _PD_BEST),
                                 ("phase_f04", _PHASE_F04)):
        rows.append({"name": f"{ctrl_name}_current",
                     **_ctrl(ctrl_base, "current")})
        rows.append({"name": f"{ctrl_name}_full_adaptive",
                     **_ctrl(ctrl_base, "full_adaptive")})
        rows.append({"name": f"{ctrl_name}_sched_floor_{floor:.2f}",
                     **_ctrl(ctrl_base, "sched_floor")})
    return rows


# ---------------------------------------------------------------------
# Phase-13 sweep: best-of-all-worlds config × kv_mix_ratio
# ---------------------------------------------------------------------


# Orthogonal winners stacked: Phase-7/8 soft_mask base + Phase-11 p90
# drift signal + Phase-12 full_adaptive (combine=replace). kv is swept.
_BEST_BASE: Dict[str, Any] = dict(
    _PD_BEST,
    drift_metric="latent_init_p90",
    combine="replace",
    inject_weight_floor=0.0,
)


def best_kv_configs(
    target_drifts: Dict[float, float],
    include_dual: bool = True,
    dual_edit_pres_ratio: float = 3.0,
    kv_grid: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Phase-13 sweep: best stack across kv_mix_ratio.

    Stacks the orthogonal wins identified across prior phases:
      - Phase 7/8: soft_mask on; channel_ls + adaptive_kv off.
      - Phase 11: drift_metric = latent_init_p90 (robust worst-token).
      - Phase 12: combine = replace (full adaptive, drops sigmoid envelope).
      - Phase 13: kv_mix_ratio swept over {0.30, 0.40, 0.50, 0.60, 0.75, 0.90}.

    Each kv row needs its own calibrated ``target_drift`` (p90 scale
    shifts with kv). ``target_drifts`` maps kv -> target_drift;
    populate via ``calibrate_drift_signals(base_cfg={**_BEST_BASE,
    'kv_mix_ratio': kv})`` for each kv.

    When ``include_dual=True``, appends two rows stacking
    ``mode='dual_objective'`` on the best config — the Phase-9
    ``dual_r3`` variant won clip_t/clip_dir within the PD family and
    its mechanism is independent of the sigmoid, so it's the only
    Phase-9 mode worth stacking. Rows land at kv=0.30 and kv=0.60
    (near-endpoint check + mid-range stack).

    Rows (8 core + 2 optional dual):
      - paper_adaedit            (anchor; original, kv=0.90)
      - pd_best                  (anchor; Phase-7/8 winner)
      - best_kv0.30 ... best_kv0.90   (6 rows, best stack × kv)
      - best_dual_kv0.30, best_dual_kv0.60  (2 optional rows)
    """
    if kv_grid is None:
        kv_grid = [0.30, 0.40, 0.50, 0.60, 0.75, 0.90]

    missing = [kv for kv in kv_grid if kv not in target_drifts]
    if missing:
        raise ValueError(
            f"target_drifts missing entries for kv={missing}; "
            "run calibrate_drift_signals(base_cfg=...) per kv first."
        )

    rows: List[Dict[str, Any]] = [
        {"name": "paper_adaedit",
         "mode": "original",
         "kv_mix_ratio": 0.9,
         "ls_ratio": 0.25,
         **_EXT_ON},
        {"name": "pd_best", **_PD_BEST},
    ]

    for kv in kv_grid:
        rows.append({
            "name": f"best_kv{kv:.2f}",
            **_BEST_BASE,
            "kv_mix_ratio": kv,
            "target_drift": target_drifts[kv],
        })

    if include_dual:
        for kv in (0.30, 0.60):
            if kv not in target_drifts:
                continue
            rows.append({
                "name": f"best_dual_kv{kv:.2f}",
                **_BEST_BASE,
                "mode": "dual_objective",
                "kv_mix_ratio": kv,
                "target_drift": target_drifts[kv],
                "edit_pres_ratio": dual_edit_pres_ratio,
            })

    return rows


# ---------------------------------------------------------------------
# Phase-15 sweep: progress_adaptive (v3) controller
# ---------------------------------------------------------------------


def progress_adaptive_configs() -> List[Dict[str, Any]]:
    """
    v3 progress_adaptive sweep: 8 configs.

    Tests the new dual-signal controller that tracks preservation drift
    AND edit velocity against per-step time-varying setpoints derived
    from the source trajectory. The goal is to beat both `paper_adaedit`
    and `pd_best` on all 9 metrics — closing the structural clip_t /
    clip_dir gap that drift-only PD controllers can't close.

    Setpoint shape:
      target_pres_profile[i]   = beta * MSE(source_traj[i], z_init)
                                 over preservation tokens
      target_edit_velocity[i]  = scale * inject_weights[i] * mean(per-step
                                 source preservation velocity)

    Effective kv:
      kv_eff = base_kv * alpha_pres * (1 - release_gain * alpha_release) * w(i)

    Anchors:
      - paper_adaedit
      - pd_best (Phase-7/8 winner)

    New variants (all use mode='progress_adaptive', kv=0.30, soft_mask
    on, drift signal latent_relative for apples-to-apples per-step
    comparison, edit signal edit_step for per-step velocity):
      - pa_default          : β=1.0, release_gain=0.5, scale=1.0
      - pa_pres_loose       : β=1.5 (tolerate more drift)
      - pa_pres_tight       : β=0.7 (clamp harder on drift)
      - pa_release_low      : release_gain=0.25 (small release)
      - pa_release_high     : release_gain=0.8 (big release)
      - pa_velocity_strong  : target_edit_velocity_scale=1.5
                              (demand more edit velocity)
    """
    _PA_BASE: Dict[str, Any] = dict(
        mode="progress_adaptive",
        kv_mix_ratio=0.30,
        ls_ratio=0.45,
        kp_p=1.5,
        kd_p=0.2,
        kp_release=1.0,
        kd_release=0.1,
        release_pres_slack=0.0,
        use_soft_mask=True,
        drift_metric="latent_relative",
        progress_edit_drift_metric="edit_step",
        combine="multiply",
        target_pres_beta=1.0,
        target_edit_velocity_scale=1.0,
        release_gain=0.5,
    )

    def _pa(name, **over):
        cfg = dict(_PA_BASE)
        cfg.update(over)
        cfg["name"] = name
        return cfg

    return [
        # Anchors
        {"name": "paper_adaedit",
         "mode": "original",
         "kv_mix_ratio": 0.9,
         "ls_ratio": 0.25,
         **_EXT_ON},
        {"name": "pd_best", **_PD_BEST},

        # New v3 variants
        _pa("pa_default"),
        _pa("pa_pres_loose", target_pres_beta=1.5),
        _pa("pa_pres_tight", target_pres_beta=0.7),
        _pa("pa_release_low", release_gain=0.25),
        _pa("pa_release_high", release_gain=0.8),
        _pa("pa_velocity_strong", target_edit_velocity_scale=1.5),
    ]


def calibrate_drift_signals(
    *,
    t5, clip, model, ae,
    n: int = 5,
    seed: int = 0,
    metrics: Optional[List[str]] = None,
    target_ratio: float = 0.8,
    inject_weight_threshold: float = 0.05,
    output_dir: str = "outputs_pie_calibration",
    num_steps: int = 15,
    run_seed: int = 42,
    inject: int = 4,
    inject_schedule: str = "sigmoid",
    include_categories: Optional[List[int]] = None,
    exclude_categories: Optional[List[int]] = (8,),
    data_root: Optional[str] = None,
    base_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Calibrate ``target_drift`` for each candidate drift signal.

    For each metric, runs a neutralized (``kp=kd=0``) controller on `n`
    PIE-Bench samples — alpha stays at ``base_alpha`` so the kv schedule
    matches an unmodified target config and we just *observe* the drift
    trajectory. Per sample, we take the median drift across steps where
    ``inject_weight > inject_weight_threshold`` (the injection window),
    then median-of-medians across samples. The returned target_drift is
    ``target_ratio * pooled_median``.

    ``base_cfg`` (default ``_PD_BEST``) is the config shape used during
    calibration — use this to calibrate under the same combine mode,
    kv_mix_ratio, mode, etc. that the downstream sweep will run. The
    metric name and ``kp=kd=0`` overrides are applied on top.

    Returns a dict mapping each metric name to its calibrated
    ``target_drift``. Values are ready to plug into
    ``drift_signal_configs`` or ``best_kv_configs``.
    """
    import json
    import statistics

    if metrics is None:
        metrics = ["latent_relative", "latent_init_cosine", "latent_init_p90"]
    if base_cfg is None:
        base_cfg = _PD_BEST

    os.makedirs(output_dir, exist_ok=True)
    calibrated: Dict[str, float] = {}

    for metric in metrics:
        cfg = dict(base_cfg)
        cfg.update(
            name=f"calib_{metric}",
            drift_metric=metric,
            kp=0.0, kd=0.0,
        )
        cfg_dir = os.path.join(output_dir, cfg["name"])

        print("\n" + "=" * 80)
        print(f"= Calibration: {metric}  (n={n}, kp=kd=0)")
        print("=" * 80)

        rows = run_pie_samples(
            t5=t5, clip=clip, model=model, ae=ae,
            n=n, seed=seed,
            config=cfg,
            data_root=data_root,
            output_dir=cfg_dir,
            include_categories=include_categories,
            exclude_categories=exclude_categories,
            num_steps=num_steps,
            run_seed=run_seed,
            inject=inject,
            inject_schedule=inject_schedule,
            csv_path=os.path.join(cfg_dir, "calib_samples.csv"),
        )

        per_sample_medians: List[float] = []
        for r in rows:
            run_dir = r.get("run_dir") or ""
            log_path = os.path.join(run_dir, "adaptive_log.json")
            if not run_dir or not os.path.exists(log_path):
                continue
            try:
                with open(log_path) as f:
                    log = json.load(f)
            except Exception:
                continue
            steps = log.get("per_step", []) if isinstance(log, dict) else log
            window = [
                float(s["drift"]) for s in steps
                if float(s.get("inject_weight", 0.0)) > inject_weight_threshold
                and s.get("drift") is not None
            ]
            if window:
                per_sample_medians.append(statistics.median(window))

        if not per_sample_medians:
            print(f"  !! no usable adaptive logs for {metric}; defaulting td=0.02")
            calibrated[metric] = 0.02
            continue

        pooled = statistics.median(per_sample_medians)
        td = target_ratio * pooled
        calibrated[metric] = td
        print(
            f"  {metric:<24}  pooled_median={pooled:.5f}  "
            f"target_drift={td:.5f}  (n_samples={len(per_sample_medians)})"
        )

    print("\nCalibrated target_drift:")
    for k, v in calibrated.items():
        print(f"  {k:<24}  {v:.5f}")
    return calibrated


def _cfg_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of cfg we echo into the aggregate CSV for each row."""
    return {
        "mode":            cfg.get("mode", "pd_adaptive"),
        "kv_mix_ratio":    cfg.get("kv_mix_ratio", ""),
        "target_drift":    cfg.get("target_drift", ""),
        "drift_metric":    cfg.get("drift_metric", ""),
        "combine":            cfg.get("combine", ""),
        "inject_weight_floor": cfg.get("inject_weight_floor", ""),
        "ls_ratio":        cfg.get("ls_ratio", ""),
        "kp":              cfg.get("kp", ""),
        "kd":              cfg.get("kd", ""),
        "use_soft_mask":   cfg.get("use_soft_mask", False),
        "use_adaptive_kv": cfg.get("use_adaptive_kv", False),
        "use_channel_ls":  cfg.get("use_channel_ls", False),
        # v2 diagnostics (empty when not set)
        "edit_pres_ratio": cfg.get("edit_pres_ratio", ""),
        "edit_drift_metric": cfg.get("edit_drift_metric", ""),
        "edit_fraction":   cfg.get("edit_fraction", ""),
        "kv_mix_edit":     cfg.get("kv_mix_edit", ""),
        "kv_mix_preserve": cfg.get("kv_mix_preserve", ""),
        "alpha_edit":      cfg.get("alpha_edit", ""),
        "td_profile":      cfg.get("td_profile", ""),
        "td_high":         cfg.get("td_high", ""),
        "td_low":          cfg.get("td_low", ""),
        "release_factor":  cfg.get("release_factor", ""),
        # v3 progress_adaptive
        "target_pres_beta": cfg.get("target_pres_beta", ""),
        "target_edit_velocity_scale": cfg.get("target_edit_velocity_scale", ""),
        "release_gain":    cfg.get("release_gain", ""),
        "kp_release":      cfg.get("kp_release", ""),
        "kd_release":      cfg.get("kd_release", ""),
        "progress_edit_drift_metric": cfg.get("progress_edit_drift_metric", ""),
    }


def _means(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-metric mean across samples that finished OK."""
    ok = [r for r in rows if r.get("status") == "ok"]
    out: Dict[str, Any] = {"n_ok": len(ok), "n_total": len(rows)}
    for m in _METRIC_ORDER:
        vals = [r[m] for r in ok if isinstance(r.get(m), (int, float))]
        out[m] = (sum(vals) / len(vals)) if vals else None
        out[f"{m}_n"] = len(vals)
    return out


def _print_cross_config_summary(summary_rows: List[Dict[str, Any]]) -> None:
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    if not ok_rows:
        print("No configs produced successful runs.")
        return

    cols = ["name", "time_min", "n_ok"] + list(_METRIC_ORDER)
    widths = {
        c: max(len(c), max(len(_fmt_cell(r.get(c))) for r in ok_rows))
        for c in cols
    }

    def _row(vals):
        return "  ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals))

    print("\n" + "=" * 120)
    print(f"PIE-BENCH CROSS-CONFIG SWEEP  ({len(ok_rows)} configs)")
    print("=" * 120)
    print(_row(cols))
    print(_row(["-" * widths[c] for c in cols]))
    for r in ok_rows:
        print(_row([_fmt_cell(r.get(c)) for c in cols]))

    print("\nBest per metric:")
    for m in _METRIC_ORDER:
        vals = [(r["name"], r.get(m)) for r in ok_rows
                if isinstance(r.get(m), (int, float))]
        if not vals:
            continue
        hib = _HIGHER_IS_BETTER[m]
        best = max(vals, key=lambda x: x[1]) if hib else min(vals, key=lambda x: x[1])
        print(f"  {m:>9} ({'↑' if hib else '↓'}): {best[0]:<25}  {best[1]:.4f}")

    # Pareto count — how many metrics each config wins
    wins: Dict[str, int] = {r["name"]: 0 for r in ok_rows}
    for m in _METRIC_ORDER:
        vals = [(r["name"], r.get(m)) for r in ok_rows
                if isinstance(r.get(m), (int, float))]
        if not vals:
            continue
        hib = _HIGHER_IS_BETTER[m]
        winner = (max if hib else min)(vals, key=lambda x: x[1])[0]
        wins[winner] += 1
    ranked = sorted(wins.items(), key=lambda kv: -kv[1])
    print("\nMetric wins (out of 9):")
    for name, w in ranked:
        if w > 0:
            print(f"  {name:<25}  {w}")


def _write_csv(summary_rows: List[Dict[str, Any]], csv_path: str) -> None:
    fixed = [
        "name", "status", "n_ok", "n_total", "time_min",
        "mode", "kv_mix_ratio", "target_drift", "drift_metric",
        "combine", "inject_weight_floor",
        "ls_ratio", "kp", "kd",
        "use_soft_mask", "use_adaptive_kv", "use_channel_ls",
        "edit_pres_ratio", "edit_drift_metric",
        "edit_fraction", "kv_mix_edit", "kv_mix_preserve", "alpha_edit",
        "td_profile", "td_high", "td_low", "release_factor",
        *_METRIC_ORDER,
        *(f"{m}_n" for m in _METRIC_ORDER),
        "per_config_csv",
    ]
    seen = set()
    fieldnames: List[str] = []
    for k in fixed:
        if k not in seen:
            fieldnames.append(k); seen.add(k)
    for r in summary_rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k); seen.add(k)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)


def run_pie_sweep(
    *,
    t5,
    clip,
    model,
    ae,
    n: int = 30,
    seed: int = 0,
    configs: Optional[List[Dict[str, Any]]] = None,
    data_root: Optional[str] = None,
    output_dir: str = "outputs_pie_sweep",
    include_categories: Optional[List[int]] = None,
    exclude_categories: Optional[List[int]] = (8,),
    num_steps: int = 15,
    run_seed: int = 42,
    inject: int = 4,
    inject_schedule: str = "sigmoid",
    csv_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run ``configs`` (default: the 13-config grid) on the same PIE-Bench
    slice and return one summary row per config.

    Each config gets its own subdirectory under ``output_dir``. The
    cross-config aggregate CSV lands at
    ``<output_dir>/sweep_pie_results.csv`` (or ``csv_path``).

    Parameters
    ----------
    n : int
        Number of samples per config. Defaults to 30 to match phase 3.
    seed : int
        Controls which samples get drawn from PIE-Bench. Defaults to 0
        so the slice matches phase 3 exactly.
    run_seed : int
        The AdaEdit pipeline seed, forwarded to every sample.
    configs : list of dict, optional
        Override the default grid. Each dict must have a ``name`` key.
    """
    configs = configs or _default_configs()
    os.makedirs(output_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(output_dir, "sweep_pie_results.csv")

    summary_rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        cfg_output_dir = os.path.join(output_dir, name)
        per_config_csv = os.path.join(cfg_output_dir, "pie_samples.csv")

        print("\n" + "#" * 100)
        print(f"# [{i}/{len(configs)}]  {name}")
        print(f"# {cfg}")
        print("#" * 100)

        t0 = time.perf_counter()
        status = "ok"
        per_sample_rows: List[Dict[str, Any]] = []
        try:
            per_sample_rows = run_pie_samples(
                t5=t5, clip=clip, model=model, ae=ae,
                n=n, seed=seed,
                config=cfg,
                data_root=data_root,
                output_dir=cfg_output_dir,
                include_categories=include_categories,
                exclude_categories=exclude_categories,
                num_steps=num_steps,
                run_seed=run_seed,
                inject=inject,
                inject_schedule=inject_schedule,
                csv_path=per_config_csv,
            )
        except Exception as e:
            status = f"failed: {type(e).__name__}: {e}"
            print(f"!! CONFIG FAILED: {status}")
        elapsed = time.perf_counter() - t0

        row: Dict[str, Any] = {
            "name": name,
            "status": status,
            "time_min": round(elapsed / 60.0, 2),
            "per_config_csv": per_config_csv,
            **_cfg_knobs(cfg),
        }
        if per_sample_rows:
            row.update(_means(per_sample_rows))
        summary_rows.append(row)

        # Incremental CSV write so a crash partway doesn't lose earlier runs.
        _write_csv(summary_rows, csv_path)

    print(f"\n✓ Wrote cross-config summary -> {csv_path}")
    _print_cross_config_summary(summary_rows)
    return summary_rows
