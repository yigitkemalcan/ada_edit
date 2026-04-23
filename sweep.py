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


if __name__ == "__main__":
    raise SystemExit(
        "sweep.py is meant to be imported from a notebook where "
        "t5/clip/model/ae are already in scope. See the docstring."
    )
