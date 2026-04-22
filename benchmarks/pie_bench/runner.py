"""
PIE-Bench random-sample runner.

Designed to be called from a Colab cell where t5 / clip / model / ae
have already been loaded, so every sample reuses the weights:

    from benchmarks.pie_bench import run_pie_samples

    results = run_pie_samples(
        n=5, seed=0,
        t5=t5, clip=clip, model=model, ae=ae,
    )

Each sample runs our pipeline once, using PIE-Bench's ground-truth
mask to compute background-preservation metrics. Per-sample results
land under <output_dir>/<key>/ (edited.jpg + metrics.json +
adaptive_log.json). An aggregate CSV is written to
<output_dir>/pie_samples.csv.

Category 8 (background replacement) is skipped by default — AdaEdit
isn't targeting that category and the user elected to omit it for
now.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

from sweep import _build_args, _METRIC_ORDER, _HIGHER_IS_BETTER, _fmt_cell
from adaedit_adaptive import run

from .loader import PIESample, sample_pie


# Phase-2 sweep winner documented in EXPERIMENTS.md — sensible default
# per-sample config for quick benchmark smoke tests.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "pie_pd_winner",
    "mode": "pd_adaptive",
    "kv_mix_ratio": 0.9,
    "target_drift": 0.02,
    "ls_ratio": 0.45,
    "kp": 1.5,
    "kd": 0.2,
}


def _run_one(
    sample: PIESample,
    cfg: Dict[str, Any],
    *,
    output_dir: str,
    num_steps: int,
    seed: int,
    inject: int,
    inject_schedule: str,
    t5, clip, model, ae,
) -> Dict[str, Any]:
    # Name each run after the PIE-Bench key so output folders are
    # traceable to the dataset entry.
    cfg_sample = {**cfg, "name": f"{cfg.get('name', 'pie')}__{sample.key}"}
    args = _build_args(
        cfg_sample,
        source_img=sample.image_path,
        source_prompt=sample.source_prompt,
        target_prompt=sample.target_prompt,
        edit_object=sample.edit_object,
        edit_type=sample.edit_type,
        output_dir=output_dir,
        num_steps=num_steps,
        seed=seed,
        inject=inject,
        inject_schedule=inject_schedule,
    )

    t0 = time.perf_counter()
    metrics: Dict[str, float] = {}
    status = "ok"
    run_dir = None
    try:
        run_dir = run(
            args,
            t5=t5, clip=clip, model=model, ae=ae,
            gt_mask=sample.mask,
        )
        metrics_fn = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_fn):
            with open(metrics_fn) as f:
                metrics = json.load(f)
    except Exception as e:
        status = f"failed: {type(e).__name__}: {e}"
        print(f"!! FAILED on {sample.key}: {status}")
    elapsed = time.perf_counter() - t0

    return {
        "key": sample.key,
        "editing_type_id": sample.editing_type_id,
        "edit_type": sample.edit_type,
        "source_prompt": sample.source_prompt,
        "target_prompt": sample.target_prompt,
        "edit_object": sample.edit_object,
        "status": status,
        "time_s": round(elapsed, 2),
        "run_dir": run_dir or "",
        **metrics,
    }


def _print_aggregate(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No samples to summarize.")
        return

    # Per-sample table
    cols = ["key", "editing_type_id", "edit_type", "time_s"] + list(_METRIC_ORDER)
    widths = {
        c: max(len(c), max(len(_fmt_cell(r.get(c))) for r in rows))
        for c in cols
    }

    def _row(vals):
        return "  ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals))

    print("\n" + "=" * 80)
    print(f"PIE-BENCH RANDOM SAMPLES  (n={len(rows)})")
    print("=" * 80)
    print(_row(cols))
    print(_row(["-" * widths[c] for c in cols]))
    for r in rows:
        print(_row([_fmt_cell(r.get(c)) for c in cols]))

    # Mean per metric across successful runs
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        print("\nMean across successful samples:")
        for m in _METRIC_ORDER:
            vals = [r[m] for r in ok_rows if isinstance(r.get(m), (int, float))]
            if vals:
                hib = _HIGHER_IS_BETTER[m]
                mean = sum(vals) / len(vals)
                print(f"  {m:>9} ({'↑' if hib else '↓'}): {mean:.4f}  "
                      f"(n={len(vals)})")

    # Per-category breakdown
    by_cat: Dict[int, List[Dict[str, Any]]] = {}
    for r in ok_rows:
        by_cat.setdefault(r["editing_type_id"], []).append(r)
    if len(by_cat) > 1:
        print("\nPer-category means:")
        for cat_id in sorted(by_cat):
            cat_rows = by_cat[cat_id]
            parts = [f"cat {cat_id} (n={len(cat_rows)})"]
            for m in ("psnr_bg", "ssim_bg", "lpips_bg", "clip_t"):
                vals = [r[m] for r in cat_rows
                        if isinstance(r.get(m), (int, float))]
                if vals:
                    parts.append(f"{m}={sum(vals)/len(vals):.4f}")
            print("  " + "  ".join(parts))


def run_pie_samples(
    *,
    t5,
    clip,
    model,
    ae,
    n: int = 5,
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
    data_root: Optional[str] = None,
    output_dir: str = "outputs_pie_samples",
    include_categories: Optional[List[int]] = None,
    exclude_categories: Optional[List[int]] = (8,),
    num_steps: int = 15,
    run_seed: int = 42,
    inject: int = 4,
    inject_schedule: str = "sigmoid",
    csv_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Draw `n` random PIE-Bench samples (seeded) and run the pipeline on
    each, using the dataset's GT mask for background-preservation
    metrics.

    Parameters
    ----------
    n : int
        Number of random samples to draw after category filtering.
    seed : int
        Controls which samples are drawn from the pool. Change this
        between calls to get a fresh slice of the benchmark.
    config : dict, optional
        The AdaEdit config dict applied to every sample (same shape
        sweep.py uses). Defaults to the phase-2 winner documented in
        EXPERIMENTS.md.
    data_root : str, optional
        Path to the cloned PnPInversion repo. Defaults to
        <project>/data/PnPInversion.
    include_categories / exclude_categories : list of int
        editing_type_id filters. By default category 8 (background) is
        excluded.
    run_seed : int
        The AdaEdit pipeline seed, forwarded unchanged for every
        sample (matches sweep.py behaviour).
    """
    cfg = dict(config) if config else dict(_DEFAULT_CONFIG)
    os.makedirs(output_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(output_dir, "pie_samples.csv")

    samples = sample_pie(
        n=n,
        seed=seed,
        root=data_root,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
    )
    print(f"Drew {len(samples)} PIE-Bench sample(s) "
          f"(seed={seed}, excluded={list(exclude_categories or [])}).")
    for s in samples:
        print(f"  {s.key}  cat={s.editing_type_id}  "
              f"{s.source_prompt!r} -> {s.target_prompt!r}")

    rows: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples, 1):
        print(f"\n=== [{i}/{len(samples)}] {sample.key} "
              f"(cat={sample.editing_type_id}, {sample.edit_type}) ===")
        rows.append(_run_one(
            sample, cfg,
            output_dir=output_dir,
            num_steps=num_steps,
            seed=run_seed,
            inject=inject,
            inject_schedule=inject_schedule,
            t5=t5, clip=clip, model=model, ae=ae,
        ))

    # --- CSV ---------------------------------------------------------
    fixed = [
        "key", "editing_type_id", "edit_type", "status", "time_s",
        "source_prompt", "target_prompt", "edit_object",
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
    print(f"\n✓ Wrote {len(rows)} sample rows -> {csv_path}")

    _print_aggregate(rows)
    return rows
