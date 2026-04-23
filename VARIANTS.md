# Adaptive Injection Variants (v2)

This document describes the five new adaptive-injection variants added
alongside the current PD winner, and how to run them. All variants
live under `src/flux/adaptive/` in new `_v2.py` files — the v1 code
paths (`pd_adaptive`, `original`, etc.) are untouched and still work.

## Why add variants

On the full 700-image PIE-Bench, the current single-objective PD
controller (mode `pd_adaptive`, with `kv_mix_ratio=0.30`,
`target_drift=0.02`, `soft_mask` on) wins every fidelity metric versus
the AdaEdit paper:

| | Ours (PD v1) | Paper | Δ |
|---|---|---|---|
| PSNR     | 20.82  | 18.57  | **+2.25 dB** |
| SSIM     | 0.717  | 0.667  | **+0.051**   |
| LPIPS    | 0.284  | 0.345  | **−17.8%**   |
| PSNR_bg  | 22.20  | 20.78  | **+1.42 dB** |
| CLIP-T   | 0.252  | 0.260  | **−3%**  ← loss |
| CLIP-dir | 0.075  | 0.100  | **−25%** ← loss |

Root cause: the PD loop has exactly one error signal (preservation
drift). It drives `kv_mix_ratio` down to minimize preservation drift,
which also suppresses the edit — there is no counter-signal that says
"I'm under-editing". The five v2 variants each introduce a different
mechanism to recover edit-alignment without sacrificing fidelity.

---

## The current winner: `pd_adaptive` (PD, v1)

Single-objective PD controller on preservation drift:

```
e_t       = drift_pres(t)  −  target_drift
control_t = kp · e_t  +  kd · (e_t − e_{t-1})
alpha_t   = clamp(base_alpha + control_t, alpha_min, alpha_max)
kv_eff    = base_kv · alpha_t · w(t)      # w from progressive schedule
```

Files: `src/flux/adaptive/{controller.py, drift.py, sampling_adaptive.py}`.
Winning config: `kv_mix_ratio=0.30`, `kp=1.5`, `kd=0.2`,
`target_drift=0.02`, `ls_ratio=0.45`, `--use_soft_mask`. This is the
`pd_best` anchor all variants are compared against.

---

## Variant A — `dual_objective`

**Idea.** Give the controller a second error signal. Measure drift in
the *edit* region as well as the preservation region. When the edit
region is stationary (under-editing), push alpha DOWN so target K/V
dominate and the edit can proceed. Classical MIMO PD.

**Control law**

```
e_pres(t) = drift_pres(t)  −  target_pres
e_edit(t) = target_edit    −  drift_edit(t)        # inverted
target_edit = edit_pres_ratio · target_pres        # scale-matched

control(t) = kp_p·e_pres + kd_p·Δe_pres
           − kp_e·e_edit − kd_e·Δe_edit            # note the minus
alpha_t    = clamp(base_alpha + control(t), alpha_min, alpha_max)
```

By default `e_pres` is floored at 0 (`--clip_pres_below_zero`, on
by default) so the controller never rewards being "too preserved" —
otherwise the preservation and edit branches could double-push alpha
below base_alpha.

**New files**
- `src/flux/adaptive/drift_v2.py` — `EditProgressMeter` with modes
  `edit_init`, `edit_step`, `edit_init_soft`, `edit_normalized`.
- `src/flux/adaptive/controller_v2.py` — `DualObjectiveController`.
- `src/flux/adaptive/sampling_adaptive_v2.py::denoise_fireflow_dual`.

**New CLI flags**
`--kp_p --kd_p --kp_e --kd_e --target_pres --edit_pres_ratio --edit_drift_metric --clip_pres_below_zero / --no_clip_pres_below_zero`.

**Log schema** (per-step extras):
`drift_pres, drift_edit, alpha, kv_mix_ratio, target_pres, target_edit, variant`.

**Hypothesis.** Directly addresses the under-editing failure mode.
Expected: CLIP-T / CLIP-dir ↑, PSNR roughly flat because the
preservation branch is still closed-loop.

---

## Variant B — `two_phase_switch`

**Idea.** Partition the denoising steps into a deterministic "edit"
phase (first `edit_fraction*T` steps) and a "preserve" phase.

- Edit phase: `alpha ← alpha_edit` (low), `kv ← kv_mix_edit` (low).
  Target K/V dominate; semantics of the edit get established.
- Preserve phase: PD controller active with `target_drift`, and base
  kv switches to `kv_mix_preserve` (high). Fidelity locks in.
- `phase_ramp_steps` linear interpolation between phases to avoid a
  feature-space discontinuity. Controller's `_prev_error` is cleared
  at the boundary to kill the derivative spike.

**Motivation.** FLUX's mu-shifted schedule concentrates semantics in
early timesteps and detail in later steps. Letting the target prompt
drive early denoising matches that prior.

**Files**: `src/flux/adaptive/sampling_adaptive_v2.py::denoise_fireflow_two_phase`.
The controller is the plain `PDController` from v1.

**New CLI flags**
`--edit_fraction --alpha_edit --kv_mix_edit --kv_mix_preserve --phase_ramp_steps`.

**Log schema** extras: `drift, alpha, kv_mix_ratio, base_kv_t, phase, variant`.

**Hypothesis.** Simpler than a MIMO controller and targets the same
failure mode. `edit_fraction` is probably edit-type-dependent.

---

## Variant C — `asymmetric_region`

**Idea.** Use different `kv_mix_ratio` values in the edit region vs
the preservation region, spatially gated by the existing `soft_mask`.

Current v1 mixing is:

```
mixed  = (1 − r) · source_k + r · target_k     # single r, everywhere
out    = m · mixed + (1 − m) · source_k        # m = soft_mask
```

So in preservation region (m → 0) the output is pure source — r has no
effect there. Variant C introduces a second ratio and mixes BOTH
regions, each with its own rate:

```
mixed_edit = (1 − r_edit)     · source_k + r_edit     · target_k
mixed_pres = (1 − r_preserve) · source_k + r_preserve · target_k
out        = m · mixed_edit + (1 − m) · mixed_pres
```

Low `r_edit` (≈0.4) gives target K/V more freedom where editing
happens; high `r_preserve` (≈0.9) keeps the background source-
dominated but allows a small amount of target leak-in, which can
improve edit coherence.

**Files.**
- `src/flux/adaptive/sampling_adaptive_v2.py::denoise_fireflow_asymmetric`
- Small guarded diff in `src/flux/modules/layers.py`: adds helper
  `apply_kv_mix_asymmetric(...)` and a branch in each KV-mix call
  site (both `DoubleStreamBlock` and `SingleStreamBlock`). Gated by
  `"kv_mix_ratio_edit" in info`; when absent, code path is identical
  to v1 (verified by syntax + pre-existing behavior preserved).

**Requires `--use_soft_mask`.** The sampler raises at runtime if it's
not set.

**New CLI flags** `--kv_mix_edit --kv_mix_preserve` (shared with B).

**Log schema** extras: `drift, alpha, kv_mix_edit, kv_mix_preserve, variant`.

**Hypothesis.** Decouples "where to edit" (soft_mask) from "how
strongly" (per-region ratio). Should win edit-alignment in the
foreground while keeping background preservation tight.

---

## Variant D — `scheduled_target`

**Idea.** Make `target_drift` time-varying. Relaxed early, tight late
— lets editing happen, locks fidelity at the end. Cheapest lever.

```
target_t = schedule(i / (T − 1))    with schedule one of:
  cosine_high_low : (1−s)·td_high + s·td_low     s = 0.5(1 − cos(π·u))
  cosine_low_high : reverse
  linear          : (1−u)·td_high + u·td_low
```

The controller is `ScheduledTargetController(PDController)` — same PD
math as v1, but `.step(drift, step_idx)` updates `self.target` each
step before delegating to `super().step()`.

**Files.** `controller_v2.py::ScheduledTargetController`,
`sampling_adaptive_v2.py::denoise_fireflow_scheduled`.

**New CLI flags** `--td_high --td_low --td_profile`.

**Log schema** extras: `drift, target_drift_t, alpha, kv_mix_ratio, variant`.

**Hypothesis.** Soft counterpart to Variant B. Still preservation-only
blind to edit progress, but a better setpoint may close some of the
CLIP-T/dir gap on its own.

---

## Variant E — `xattn_boost`

**Idea.** Use a perceptually-grounded proxy for "am I editing the
right concept?" — the image-tokens → edit-word cross-attention mean.
When this signal drops AND we have preservation budget, temporarily
reduce alpha (so target K/V dominate) to let the model re-attend.

```
xattn_t  = mean(attn_map[:, :, image_tokens, edit_word])    # per step
e_xattn  = target_xattn − xattn_t
e_pres   = drift_pres(t) − controller.target

if e_xattn > xattn_release_threshold and e_pres < 0:
    alpha_t ← alpha_t · release_factor                   # e.g. 0.7
```

**Files.**
- `sampling_adaptive_v2.py::denoise_fireflow_xattn`
- 3-line capture hook in `layers.py` (`_maybe_capture_xattn`) that
  fires only when `info["capture_xattn_to_editword"]` is True and
  `info["id"] == 18` and `not info["second_order"]` — at most one
  write per step. The sampler sets the flag on entry and clears it
  on exit. Falls back to source-side `key_word_index` if the target
  prompt doesn't contain the edit object.

**New CLI flags**
`--target_xattn --xattn_release_threshold --release_factor`.

**Log schema** extras: `drift, alpha, kv_mix_ratio, xattn_edit_score, released, variant`.

**Hypothesis.** Directly addresses CLIP-T by closing the loop on
semantic alignment. `target_xattn` will probably need calibration
per concept; for the sweep we pick a reasonable default and let the
sweep's grid decide.

---

## Side-by-side comparison

| | Signals used | Controller | New CLI flags (variant-specific) | layers.py diff | Expected CLIP-dir gain | Added log fields |
|---|---|---|---|---|---|---|
| **PD (v1 winner)** | preservation drift | PDController | — | none | baseline | drift, alpha, kv_mix_ratio |
| **A dual_objective** | pres + edit drift | DualObjectiveController (MIMO PD) | `--kp_p --kd_p --kp_e --kd_e --target_pres --edit_pres_ratio --edit_drift_metric` | none | HIGH — explicit edit counter-force | drift_pres, drift_edit, alpha, target_pres, target_edit |
| **B two_phase_switch** | preservation drift (preserve phase only) | PDController + schedule | `--edit_fraction --alpha_edit --kv_mix_edit --kv_mix_preserve --phase_ramp_steps` | none | HIGH — early edit freedom | drift, alpha, base_kv_t, phase |
| **C asymmetric_region** | preservation drift + soft_mask | PDController | `--kv_mix_edit --kv_mix_preserve` (req. `--use_soft_mask`) | guarded branch (5 lines × 2 sites + helper) | HIGH — decouples where from how-strongly | drift, alpha, kv_mix_edit, kv_mix_preserve |
| **D scheduled_target** | preservation drift | ScheduledTargetController | `--td_high --td_low --td_profile` | none | MODERATE — soft version of B | drift, target_drift_t, alpha |
| **E xattn_boost** | pres drift + xattn | PDController + release rule | `--target_xattn --xattn_release_threshold --release_factor` | 3-line guarded hook | MODERATE — perceptual proxy | drift, alpha, xattn_edit_score, released |

---

## How to run each variant

All commands assume models have been pre-loaded; the CLI form is:

```bash
# A — dual_objective (MIMO PD)
python adaedit_adaptive.py \
  -i source.jpg -sp "A photo of a cat" -tp "A photo of a dog" \
  --edit_object cat --mode dual_objective \
  --kv_mix_ratio 0.30 --ls_ratio 0.45 \
  --kp_p 1.5 --kd_p 0.2 --kp_e 1.0 --kd_e 0.1 \
  --target_pres 0.02 --edit_pres_ratio 3.0 \
  --edit_drift_metric edit_init --use_soft_mask

# B — two_phase_switch
python adaedit_adaptive.py \
  -i source.jpg -sp "..." -tp "..." --edit_object cat \
  --mode two_phase_switch \
  --edit_fraction 0.4 --alpha_edit 0.3 \
  --kv_mix_edit 0.45 --kv_mix_preserve 0.9 \
  --phase_ramp_steps 2 \
  --kp 1.5 --kd 0.2 --target_drift 0.02 \
  --ls_ratio 0.45 --use_soft_mask

# C — asymmetric_region (requires soft_mask)
python adaedit_adaptive.py \
  -i source.jpg -sp "..." -tp "..." --edit_object cat \
  --mode asymmetric_region \
  --kv_mix_edit 0.45 --kv_mix_preserve 0.9 \
  --kp 1.5 --kd 0.2 --target_drift 0.02 \
  --ls_ratio 0.45 --use_soft_mask

# D — scheduled_target
python adaedit_adaptive.py \
  -i source.jpg -sp "..." -tp "..." --edit_object cat \
  --mode scheduled_target \
  --td_high 0.08 --td_low 0.015 --td_profile cosine_high_low \
  --kv_mix_ratio 0.30 --ls_ratio 0.45 \
  --kp 1.5 --kd 0.2 --use_soft_mask

# E — xattn_boost
python adaedit_adaptive.py \
  -i source.jpg -sp "..." -tp "..." --edit_object cat \
  --mode xattn_boost \
  --target_xattn 0.08 --xattn_release_threshold 0.02 --release_factor 0.7 \
  --kv_mix_ratio 0.30 --ls_ratio 0.45 \
  --kp 1.5 --kd 0.2 --target_drift 0.02 --use_soft_mask
```

Every run writes `edited.jpg`, `adaptive_log.json` (per-step trace of
drift, alpha, and variant-specific fields), and `metrics.json` to
`outputs_adaptive/<run_name>/`.

---

## PIE-Bench sweep (decide the winner)

A 70-sample PIE-Bench sweep comparing all 5 variants against the
paper baseline and the v1 PD winner lives in `sweep_pie_bench.py::v2_configs()`.
In a notebook with models loaded:

```python
from sweep_pie_bench import run_pie_sweep, v2_configs

results = run_pie_sweep(
    t5=t5, clip=clip, model=model, ae=ae,
    n=70, seed=0, exclude_categories=(8,),
    configs=v2_configs(),
    output_dir="outputs_v2_sweep",
)
```

Grid (~21 configs, ~5 h on a single A100 incl. CLIP/LPIPS overhead):

| anchors | A dual | B phase | C asym | D sched | E xattn |
|---|---|---|---|---|---|
| baseline, paper_adaedit, pd_best | 4 configs (ratio ∈ {2,3,5}, one `edit_normalized`) | 4 configs (edit_fraction ∈ {0.3,0.4,0.5}, one kv-high) | 4 configs (r_edit × r_preserve grid) | 3 configs (profile ∈ {high_low, low_high, linear}) | 2 configs (release_factor ∈ {0.7, 0.5}) |

Output:
- `outputs_v2_sweep/<config_name>/<sample_key>/{edited.jpg, metrics.json, adaptive_log.json}` — per-sample.
- `outputs_v2_sweep/<config_name>/pie_samples.csv` — per-config detail.
- `outputs_v2_sweep/sweep_pie_results.csv` — aggregate, one row per config, directly comparable (same 70 samples, same seed).

**Winner selection.** A config wins if it strictly dominates
`paper_adaedit` on all six metrics (PSNR, SSIM, LPIPS, PSNR_bg,
CLIP-T, CLIP-dir). Tiebreak by CLIP-dir (the deepest current
deficit), then PSNR_bg. The winner then runs on the full 688-sample
PIE-Bench vs the paper baseline to confirm.
