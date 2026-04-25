"""
Small sweep utility for AdaEdit-Adaptive experiments.

Designed to be called from a Colab cell where t5/clip/model/ae have
already been loaded, so every run reuses the weights. Writes one CSV
summarizing all runs plus the usual per-run folders.

Usage (Colab):

    from sweep import run_sweep

    results = run_sweep(
        source_img="examples/horse.jpg",
        source_prompt="A photo of a horse",
        target_prompt="A photo of a toy horse",
        edit_object="horse",
        t5=t5, clip=clip, model=model, ae=ae,
        output_dir="outputs_sweep_horse",
    )

Return value is a list of dicts (one per run). A CSV is also written to
<output_dir>/sweep_results.csv and a pretty summary is printed.

You can override the default grid by passing `configs=[...]`.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

from adaedit_adaptive import build_parser, run


# ---------------------------------------------------------------------
# Default grid — Tier 1 (hyperparam sweep) + Tier 2 (feature toggles)
# ---------------------------------------------------------------------

def _default_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    # --- Tier 1: hyperparameter grid (pd_adaptive) ------------------
    for kv in (0.6, 0.75, 0.9):
        for td in (0.02, 0.05):
            for ls in (0.25, 0.45):
                configs.append({
                    "name": f"t1_kv{kv}_td{td}_ls{ls}",
                    "mode": "pd_adaptive",
                    "kv_mix_ratio": kv,
                    "target_drift": td,
                    "ls_ratio": ls,
                })

    # --- Tier 2: feature toggles (on top of middle-of-grid defaults)
    t2_base = dict(kv_mix_ratio=0.75, target_drift=0.05, ls_ratio=0.25)
    configs.append({
        "name": "t2_soft_mask",
        "mode": "pd_adaptive",
        **t2_base,
        "use_soft_mask": True,
        "soft_mask_gamma": 5.0,
    })
    configs.append({
        "name": "t2_adaptive_kv",
        "mode": "pd_adaptive",
        **t2_base,
        "use_adaptive_kv": True,
    })
    configs.append({
        "name": "t2_channel_ls",
        "mode": "pd_adaptive",
        **t2_base,
        "use_channel_ls": True,
    })

    # --- Baselines for reference -----------------------------------
    configs.append({
        "name": "baseline_original",
        "mode": "original",
        "kv_mix_ratio": 0.9,
        "ls_ratio": 0.25,
    })
    configs.append({
        "name": "baseline_pd_default",
        "mode": "pd_adaptive",
        "kv_mix_ratio": 0.9,
        "target_drift": 0.05,
        "ls_ratio": 0.25,
    })
    return configs


# ---------------------------------------------------------------------
# Build argparse Namespace from a config dict
# ---------------------------------------------------------------------

def _build_args(
    cfg: Dict[str, Any],
    *,
    source_img: str,
    source_prompt: str,
    target_prompt: str,
    edit_object: str,
    edit_type: str,
    output_dir: str,
    num_steps: int,
    seed: int,
    inject: int,
    inject_schedule: str,
):
    cli = [
        "--source_img", source_img,
        "--source_prompt", source_prompt,
        "--target_prompt", target_prompt,
        "--edit_object", edit_object,
        "--edit_type", edit_type,
        "--num_steps", str(num_steps),
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--inject", str(inject),
        "--inject_schedule", inject_schedule,
        "--run_name", cfg["name"],
        "--mode", cfg.get("mode", "pd_adaptive"),
    ]
    # scalar knobs
    for k, flag in [
        ("kv_mix_ratio",   "--kv_mix_ratio"),
        ("target_drift",   "--target_drift"),
        ("ls_ratio",       "--ls_ratio"),
        ("soft_mask_gamma","--soft_mask_gamma"),
        ("kp",             "--kp"),
        ("kd",             "--kd"),
        ("ki",             "--ki"),
        ("base_alpha",     "--base_alpha"),
        ("alpha_min",      "--alpha_min"),
        ("alpha_max",      "--alpha_max"),
        ("drift_metric",   "--drift_metric"),
        ("combine",        "--combine"),
        ("inject_weight_floor", "--inject_weight_floor"),
        # v2 knobs (only relevant for v2 modes, ignored otherwise)
        ("kp_p",            "--kp_p"),
        ("kd_p",            "--kd_p"),
        ("kp_e",            "--kp_e"),
        ("kd_e",            "--kd_e"),
        ("target_pres",     "--target_pres"),
        ("edit_pres_ratio", "--edit_pres_ratio"),
        ("edit_drift_metric","--edit_drift_metric"),
        ("edit_fraction",   "--edit_fraction"),
        ("alpha_edit",      "--alpha_edit"),
        ("kv_mix_edit",     "--kv_mix_edit"),
        ("kv_mix_preserve", "--kv_mix_preserve"),
        ("phase_ramp_steps","--phase_ramp_steps"),
        ("td_high",         "--td_high"),
        ("td_low",          "--td_low"),
        ("td_profile",      "--td_profile"),
        ("target_xattn",    "--target_xattn"),
        ("xattn_release_threshold", "--xattn_release_threshold"),
        ("release_factor",  "--release_factor"),
        ("channel_ls_temp", "--channel_ls_temp"),
        # v3 progress_adaptive
        ("target_pres_beta", "--target_pres_beta"),
        ("target_edit_velocity_scale", "--target_edit_velocity_scale"),
        ("release_gain",    "--release_gain"),
        ("kp_release",      "--kp_release"),
        ("kd_release",      "--kd_release"),
        ("release_pres_slack", "--release_pres_slack"),
        ("progress_edit_drift_metric", "--progress_edit_drift_metric"),
    ]:
        if k in cfg:
            cli += [flag, str(cfg[k])]
    # boolean toggles
    for k, flag in [
        ("use_soft_mask",   "--use_soft_mask"),
        ("use_adaptive_kv", "--use_adaptive_kv"),
        ("use_channel_ls",  "--use_channel_ls"),
    ]:
        if cfg.get(k, False):
            cli.append(flag)
    # special-case: clip_pres_below_zero defaults True; explicit False turns it off.
    if cfg.get("clip_pres_below_zero", True) is False:
        cli.append("--no_clip_pres_below_zero")
    return build_parser().parse_args(cli)


# ---------------------------------------------------------------------
# Pretty print helper
# ---------------------------------------------------------------------

_METRIC_ORDER = (
    "psnr", "ssim", "lpips",
    "psnr_bg", "ssim_bg", "lpips_bg",
    "clip_i", "clip_t", "clip_dir",
)
# True means "higher is better". LPIPS variants are the only ↓.
_HIGHER_IS_BETTER = {
    "psnr": True, "ssim": True, "lpips": False,
    "psnr_bg": True, "ssim_bg": True, "lpips_bg": False,
    "clip_i": True, "clip_t": True, "clip_dir": True,
}


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No runs to summarize.")
        return

    cols = ["name", "time_s"] + list(_METRIC_ORDER)
    widths = {c: max(len(c), max(
        len(_fmt_cell(r.get(c))) for r in rows
    )) for c in cols}

    def _row(vals):
        return "  ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals))

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(_row(cols))
    print(_row(["-" * widths[c] for c in cols]))
    for r in rows:
        print(_row([_fmt_cell(r.get(c)) for c in cols]))

    # Winner per metric
    print("\nBest per metric:")
    for m in _METRIC_ORDER:
        vals = [(r["name"], r.get(m)) for r in rows if isinstance(r.get(m), (int, float))]
        if not vals:
            continue
        hib = _HIGHER_IS_BETTER[m]
        best = max(vals, key=lambda x: x[1]) if hib else min(vals, key=lambda x: x[1])
        print(f"  {m:>9} ({'↑' if hib else '↓'}): {best[0]}  ({best[1]:.4f})")


def _fmt_cell(v) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_sweep(
    *,
    source_img: str,
    source_prompt: str,
    target_prompt: str,
    edit_object: str,
    t5,
    clip,
    model,
    ae,
    output_dir: str = "outputs_sweep",
    configs: Optional[List[Dict[str, Any]]] = None,
    num_steps: int = 15,
    seed: int = 42,
    inject: int = 4,
    inject_schedule: str = "sigmoid",
    edit_type: str = "change",
    csv_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run a grid of configs with pre-loaded models, then summarize.

    Every run writes its own <output_dir>/<name>/ folder with
    edited.jpg + metrics.json + adaptive_log.json. The aggregate CSV
    is written to <output_dir>/sweep_results.csv (or `csv_path`).
    """
    configs = configs or _default_configs()
    os.makedirs(output_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(output_dir, "sweep_results.csv")

    rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] {cfg['name']} ===")
        args = _build_args(
            cfg,
            source_img=source_img,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_object=edit_object,
            edit_type=edit_type,
            output_dir=output_dir,
            num_steps=num_steps,
            seed=seed,
            inject=inject,
            inject_schedule=inject_schedule,
        )

        t0 = time.perf_counter()
        metrics: Dict[str, float] = {}
        run_dir = None
        status = "ok"
        try:
            run_dir = run(args, t5=t5, clip=clip, model=model, ae=ae)
            metrics_fn = os.path.join(run_dir, "metrics.json")
            if os.path.exists(metrics_fn):
                with open(metrics_fn) as f:
                    metrics = json.load(f)
        except Exception as e:
            status = f"failed: {type(e).__name__}: {e}"
            print(f"!! FAILED: {status}")
        elapsed = time.perf_counter() - t0

        rows.append({
            "name": cfg["name"],
            "status": status,
            "time_s": round(elapsed, 2),
            "run_dir": run_dir or "",
            "mode": cfg.get("mode", "pd_adaptive"),
            "kv_mix_ratio":   cfg.get("kv_mix_ratio", ""),
            "target_drift":   cfg.get("target_drift", ""),
            "ls_ratio":       cfg.get("ls_ratio", ""),
            "use_soft_mask":  cfg.get("use_soft_mask", False),
            "use_adaptive_kv":cfg.get("use_adaptive_kv", False),
            "use_channel_ls": cfg.get("use_channel_ls", False),
            **metrics,
        })

    # --- write CSV (union of all seen keys, with a stable prefix) ---
    fixed = [
        "name", "status", "time_s",
        "mode", "kv_mix_ratio", "target_drift", "ls_ratio",
        "use_soft_mask", "use_adaptive_kv", "use_channel_ls",
        *_METRIC_ORDER,
        "run_dir",
    ]
    seen = set()
    fieldnames: List[str] = []
    for k in fixed:
        if k not in seen:
            fieldnames.append(k); seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k); seen.add(k)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Wrote {len(rows)} rows -> {csv_path}")

    _print_summary(rows)
    return rows


# ---------------------------------------------------------------------
# Follow-up sweep (phase 2): refine around a known winner
# ---------------------------------------------------------------------

def followup_configs(
    *,
    kv_mix_ratio: float = 0.9,
    target_drift: float = 0.02,
    winner_ls: float = 0.45,
    ls_extension: tuple = (0.55, 0.65, 0.75),
    kp_grid: tuple = (0.5, 1.0, 1.5),
    kd_grid: tuple = (0.1, 0.2, 0.4),
) -> List[Dict[str, Any]]:
    """
    Build a config list that:
      1. Extends `ls_ratio` beyond the previous winner (0.45) to see
         where the fg-noise benefit tops out.
      2. Sweeps the PD controller gains (kp, kd) at the winning
         `ls_ratio` / `kv_mix_ratio` / `target_drift`.

    The default kp/kd grid includes the previous defaults (1.0, 0.2)
    as its center cell, so the winner is re-run as part of the grid
    (useful as a sanity reference — should reproduce prior numbers).
    """
    cfgs: List[Dict[str, Any]] = []

    # (1) ls_ratio extension (keeps kp=1.0, kd=0.2 defaults)
    for ls in ls_extension:
        cfgs.append({
            "name": f"f_ls{ls}",
            "mode": "pd_adaptive",
            "kv_mix_ratio": kv_mix_ratio,
            "target_drift": target_drift,
            "ls_ratio": ls,
        })

    # (2) kp/kd grid at the winning ls_ratio
    for kp in kp_grid:
        for kd in kd_grid:
            cfgs.append({
                "name": f"f_kp{kp}_kd{kd}",
                "mode": "pd_adaptive",
                "kv_mix_ratio": kv_mix_ratio,
                "target_drift": target_drift,
                "ls_ratio": winner_ls,
                "kp": kp,
                "kd": kd,
            })
    return cfgs


def run_followup_sweep(
    *,
    source_img: str,
    source_prompt: str,
    target_prompt: str,
    edit_object: str,
    t5,
    clip,
    model,
    ae,
    output_dir: str = "outputs_sweep_followup",
    # winning config from phase 1
    winner_kv: float = 0.9,
    winner_td: float = 0.02,
    winner_ls: float = 0.45,
    # grid knobs (override to customize)
    ls_extension: tuple = (0.55, 0.65, 0.75),
    kp_grid: tuple = (0.5, 1.0, 1.5),
    kd_grid: tuple = (0.1, 0.2, 0.4),
    num_steps: int = 15,
    seed: int = 42,
    inject: int = 4,
    inject_schedule: str = "sigmoid",
    edit_type: str = "change",
    csv_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Phase-2 sweep: refine around the phase-1 winner.

    Runs (by default):
      - 3 configs extending ls_ratio to {0.55, 0.65, 0.75}
      - 9 configs over kp ∈ {0.5, 1.0, 1.5} × kd ∈ {0.1, 0.2, 0.4}
    Total: 12 runs (≈2–3 min on T4/A100 after models are loaded).
    """
    configs = followup_configs(
        kv_mix_ratio=winner_kv,
        target_drift=winner_td,
        winner_ls=winner_ls,
        ls_extension=ls_extension,
        kp_grid=kp_grid,
        kd_grid=kd_grid,
    )
    return run_sweep(
        source_img=source_img,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        edit_object=edit_object,
        t5=t5, clip=clip, model=model, ae=ae,
        output_dir=output_dir,
        configs=configs,
        num_steps=num_steps,
        seed=seed,
        inject=inject,
        inject_schedule=inject_schedule,
        edit_type=edit_type,
        csv_path=csv_path,
    )


# ---------------------------------------------------------------------
# v2 smoke test — run pd_best + all 5 new variants on one image
# ---------------------------------------------------------------------

def smoke_v2_configs() -> List[Dict[str, Any]]:
    """
    The 6 configs used by smoke_test_v2: pd_best anchor + one
    representative config for each of the 5 v2 variants. These mirror
    the middle-of-the-road defaults from sweep_pie_bench.v2_configs(),
    so if a variant crashes on the notebook image it will also crash
    in the full PIE-Bench sweep.
    """
    # All variants start from the current v1 winner's non-mode knobs.
    base: Dict[str, Any] = dict(
        kv_mix_ratio=0.30,
        target_drift=0.02,
        ls_ratio=0.45,
        kp=1.5,
        kd=0.2,
        use_soft_mask=True,
    )
    return [
        {"name": "pd_best", "mode": "pd_adaptive", **base},
        {
            "name": "dual_r3",
            "mode": "dual_objective",
            **base,
            "kp_p": 1.5, "kd_p": 0.2, "kp_e": 1.0, "kd_e": 0.1,
            "target_pres": 0.02, "edit_pres_ratio": 3.0,
            "edit_drift_metric": "edit_init",
        },
        {
            "name": "phase_f04",
            "mode": "two_phase_switch",
            **base,
            "edit_fraction": 0.4, "alpha_edit": 0.3,
            "kv_mix_edit": 0.45, "kv_mix_preserve": 0.9,
            "phase_ramp_steps": 2,
        },
        {
            "name": "asym_45_90",
            "mode": "asymmetric_region",
            **base,
            "kv_mix_edit": 0.45, "kv_mix_preserve": 0.9,
        },
        {
            "name": "sched_hl",
            "mode": "scheduled_target",
            **base,
            "td_high": 0.08, "td_low": 0.015,
            "td_profile": "cosine_high_low",
        },
        {
            "name": "xattn_07",
            "mode": "xattn_boost",
            **base,
            "target_xattn": 0.08,
            "xattn_release_threshold": 0.02,
            "release_factor": 0.7,
        },
    ]


def _check_v2_log(run_dir: str, mode: str, num_steps: int) -> Dict[str, Any]:
    """
    Light inspection of adaptive_log.json for a single run. Returns a
    dict of diagnostic flags the smoke-test printer uses.
    """
    info: Dict[str, Any] = {"mode": mode, "log_ok": False}
    log_fn = os.path.join(run_dir, "adaptive_log.json")
    if not os.path.exists(log_fn):
        info["error"] = "missing adaptive_log.json"
        return info
    try:
        with open(log_fn) as f:
            payload = json.load(f)
    except Exception as e:
        info["error"] = f"unreadable log: {e}"
        return info
    per_step = payload.get("per_step", [])
    info["n_steps"] = len(per_step)
    info["steps_match"] = len(per_step) == num_steps

    # Fields expected per variant (presence check on step 0).
    expected = {
        "pd_adaptive":       {"drift", "alpha", "kv_mix_ratio"},
        "dual_objective":    {"drift_pres", "drift_edit", "alpha",
                              "target_pres", "target_edit"},
        "two_phase_switch":  {"drift", "alpha", "base_kv_t", "phase"},
        "asymmetric_region": {"drift", "alpha",
                              "kv_mix_edit", "kv_mix_preserve"},
        "scheduled_target":  {"drift", "target_drift_t", "alpha"},
        "xattn_boost":       {"drift", "alpha", "xattn_edit_score",
                              "released"},
    }.get(mode, set())
    if per_step and expected:
        keys0 = set(per_step[0].keys())
        missing = expected - keys0
        info["missing_fields"] = sorted(missing)
    info["log_ok"] = True
    return info


def smoke_test_v2(
    *,
    source_img: str,
    source_prompt: str,
    target_prompt: str,
    edit_object: str,
    t5,
    clip,
    model,
    ae,
    edit_type: str = "change",
    output_dir: str = "outputs_v2_smoke",
    num_steps: int = 15,
    seed: int = 42,
    inject: int = 4,
    inject_schedule: str = "sigmoid",
    configs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the v1 PD winner plus all 5 v2 variants on a single image, then
    print a compact per-variant report.

    Parameters
    ----------
    source_img, source_prompt, target_prompt, edit_object : str
        The editing task (same meaning as in adaedit_adaptive.py).
    t5, clip, model, ae : torch modules
        Pre-loaded models, passed through to ``run_sweep``.
    edit_type : str
        'change' / 'add' / 'remove' / 'style'. Default 'change'.
    output_dir : str
        Where each run writes its <run_name>/ folder.
    configs : list of dict, optional
        Override the 6-config default. Each dict needs a 'name' key.

    Returns
    -------
    list[dict]
        One row per config with metrics + status + diagnostic flags
        from adaptive_log.json.

    Notes
    -----
    After running, paste the printed summary block back into chat — the
    per-variant flags (missing fields, step count, alpha min/max) are
    enough to tell whether each variant's controller and log schema
    are behaving correctly.
    """
    configs = configs or smoke_v2_configs()
    print(f"Smoke-testing {len(configs)} configs on {source_img}")
    print(f"  source: {source_prompt!r}")
    print(f"  target: {target_prompt!r}")
    print(f"  edit:   {edit_object!r} ({edit_type})")
    print("-" * 72)

    rows = run_sweep(
        source_img=source_img,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        edit_object=edit_object,
        t5=t5, clip=clip, model=model, ae=ae,
        output_dir=output_dir,
        configs=configs,
        num_steps=num_steps,
        seed=seed,
        inject=inject,
        inject_schedule=inject_schedule,
        edit_type=edit_type,
    )

    # Augment each row with log-schema diagnostics.
    for row, cfg in zip(rows, configs):
        run_dir = row.get("run_dir", "")
        mode = cfg.get("mode", "pd_adaptive")
        row["log_check"] = (
            _check_v2_log(run_dir, mode, num_steps) if run_dir else
            {"mode": mode, "log_ok": False, "error": "no run_dir"}
        )
        # Also surface per-step alpha range — useful sanity indicator.
        try:
            with open(os.path.join(run_dir, "adaptive_log.json")) as f:
                log = json.load(f)
            alphas = [s["alpha"] for s in log.get("per_step", [])
                      if isinstance(s.get("alpha"), (int, float))]
            row["alpha_min"] = round(min(alphas), 4) if alphas else None
            row["alpha_max"] = round(max(alphas), 4) if alphas else None
        except Exception:
            row["alpha_min"] = row["alpha_max"] = None

    _print_v2_smoke_summary(rows)
    return rows


def _print_v2_smoke_summary(rows: List[Dict[str, Any]]) -> None:
    """Compact table + per-variant diagnostics for smoke_test_v2."""
    print("\n" + "=" * 100)
    print("V2 SMOKE TEST SUMMARY")
    print("=" * 100)
    hdr = f"{'name':<14} {'mode':<20} {'status':<8} {'psnr':>6} {'ssim':>6} {'lpips':>6} " \
          f"{'clip_t':>7} {'clip_dir':>8} {'α_min':>6} {'α_max':>6} {'steps':>5} {'log':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lc = r.get("log_check", {})
        missing = lc.get("missing_fields", [])
        log_flag = "OK" if lc.get("log_ok") and not missing else "FAIL"
        step_str = f"{lc.get('n_steps','-')}/{15 if lc.get('steps_match') else '?'}"
        line = (
            f"{r.get('name','-'):<14} "
            f"{r.get('mode','-'):<20} "
            f"{('OK' if r.get('status') == 'ok' else 'FAIL'):<8} "
            f"{_fmt_cell(r.get('psnr')):>6} "
            f"{_fmt_cell(r.get('ssim')):>6} "
            f"{_fmt_cell(r.get('lpips')):>6} "
            f"{_fmt_cell(r.get('clip_t')):>7} "
            f"{_fmt_cell(r.get('clip_dir')):>8} "
            f"{_fmt_cell(r.get('alpha_min')):>6} "
            f"{_fmt_cell(r.get('alpha_max')):>6} "
            f"{step_str:>5} "
            f"{log_flag:>4}"
        )
        print(line)

    # Per-variant missing-field warnings (only if any).
    for r in rows:
        lc = r.get("log_check", {})
        missing = lc.get("missing_fields", [])
        if missing:
            print(f"  ! {r['name']} missing log fields: {missing}")
        if not r.get("status", "").startswith("ok"):
            print(f"  ! {r['name']} run failed: {r.get('status')}")


if __name__ == "__main__":
    raise SystemExit(
        "sweep.py is meant to be imported from a notebook where "
        "t5/clip/model/ae are already in scope. See the docstring."
    )
