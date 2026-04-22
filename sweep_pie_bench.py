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


def _cfg_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of cfg we echo into the aggregate CSV for each row."""
    return {
        "mode":            cfg.get("mode", "pd_adaptive"),
        "kv_mix_ratio":    cfg.get("kv_mix_ratio", ""),
        "target_drift":    cfg.get("target_drift", ""),
        "ls_ratio":        cfg.get("ls_ratio", ""),
        "kp":              cfg.get("kp", ""),
        "kd":              cfg.get("kd", ""),
        "use_soft_mask":   cfg.get("use_soft_mask", False),
        "use_adaptive_kv": cfg.get("use_adaptive_kv", False),
        "use_channel_ls":  cfg.get("use_channel_ls", False),
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
        "mode", "kv_mix_ratio", "target_drift", "ls_ratio", "kp", "kd",
        "use_soft_mask", "use_adaptive_kv", "use_channel_ls",
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
