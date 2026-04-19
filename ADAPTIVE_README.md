# AdaEdit-Adaptive (experimental research variant)

An experimental fork of the AdaEdit injection path that replaces the
hand-designed injection schedule `w(t)` with a **closed-loop PD / PID
controller** driven by a **drift signal** measured in preservation
(non-edit) regions.

**The original AdaEdit pipeline is untouched.** This README describes
only the added experimental path.

## Motivation

AdaEdit applies `delta_eff(i) = delta_base * w(i)` where `w(i)` is a
progressive schedule (binary / sigmoid / cosine / linear). The
schedule is open-loop — it ignores how the current trajectory is
actually behaving.

We replace this with a feedback loop:

```
drift_t      = MSE(x_t, ref) on preservation tokens
error_t      = drift_t - target_drift
alpha_t      = clamp(base_alpha + kp*error + ki*I + kd*derivative,
                     alpha_min, alpha_max)
delta_eff(i) = delta_base * alpha_t * w(i)        # 'multiply'
             = delta_base * alpha_t               # 'replace'
```

When drift grows too large (structure is being lost), the controller
raises `alpha_t`, which increases source KV injection and restores
preservation. When drift is small, `alpha_t` drops so editing proceeds
more freely.

## What was added

New files (nothing else in the repo was modified):

```
src/flux/adaptive/
├── __init__.py               # package exports
├── controller.py             # PDController, PIDController, make_controller
├── drift.py                  # DriftMeter + pluggable drift modes
├── metrics.py                # optional light metrics (numpy/PIL only)
└── sampling_adaptive.py      # denoise_fireflow_adaptive (target pass only)

adaedit_adaptive.py           # new experiment runner / CLI entry point
ADAPTIVE_README.md            # this file
```

Files explicitly **not** touched:

- `adaedit.py` (original CLI)
- `api.py` (original Python API)
- `src/flux/sampling.py` (original sampler + progressive schedule)
- `src/flux/model.py`, `src/flux/modules/layers.py` (KV-Mix, masks,
  soft-mask, adaptive-KV all unchanged)
- `src/flux/util.py`, autoencoder, conditioner

The adaptive runner calls the original `denoise_fireflow` for the
inversion pass and for the Phase-2 Latents-Shift path, and only
replaces the Phase-3 target pass when a non-`original` mode is
selected.

## How the adaptive path integrates

Integration is sampler-level and minimally invasive:

1. Inversion runs through the original `denoise_fireflow`. This
   produces `z_inv` and populates `info['indices']` (binary) or
   `info['soft_mask']` (soft) exactly as before.
2. Latents-Shift runs through the original `latents_shift` /
   `channel_selective_latents_shift`.
3. The **target pass** uses `denoise_fireflow_adaptive`, which at each
   step:
   - Computes `drift_t` via `DriftMeter.update(img, z_init, indices,
     soft_mask)`.
   - Steps the controller to produce `alpha_t`.
   - Writes `info['kv_mix_ratio'] = clamp(base_kv * alpha_t *
     w_i_or_1, 0, 1)`.
   - Calls the model exactly as `denoise_fireflow` does.

Because the existing `DoubleStreamBlock` / `SingleStreamBlock` read
`info['kv_mix_ratio']` every step, no layer code had to change. The
base value is restored on the `info` dict at the end of the pass.

## Modes

Selected via `--mode`:

| Mode | Controller | Behaviour |
|------|------------|-----------|
| `original` | none | Vanilla AdaEdit. Calls original sampler. |
| `scheduled_fixed` | none | Alias of `original`. Kept for ablation. |
| `fixed_soft` | constant | Constant `alpha = base_alpha` (no feedback). A non-adaptive baseline between original and PD. |
| `pd_adaptive` | PD | Our main proposed mode. |
| `pid_adaptive` | PID | PD + clipped integral term. |

## Drift metrics (`--drift_metric`)

All operate on the packed token sequence (`[B, L, C]`) the model is
running on, restricted to preservation tokens (complement of the
edit mask).

- `latent_init` — MSE against `z_init` using binary edit indices.
- `latent_init_soft` — MSE against `z_init` using `(1 - soft_mask)`.
- `latent_step` — MSE against previous step's latent (drift velocity).
- `latent_combined` — mean of `latent_init` and `latent_step`.

`DriftMeter.reset()` is called at the start of each target pass, so
nothing leaks between runs.

## Controller hyperparameters

Passed on the CLI:

| Flag | Default | Meaning |
|------|---------|---------|
| `--kp` | 1.0 | Proportional gain |
| `--ki` | 0.1 | Integral gain (PID only) |
| `--kd` | 0.2 | Derivative gain |
| `--target_drift` | 0.05 | Reference drift level we want to hold |
| `--base_alpha` | 1.0 | Bias applied before P/I/D contributions |
| `--alpha_min` | 0.2 | Lower clamp on `alpha_t` |
| `--alpha_max` | 1.2 | Upper clamp on `alpha_t` |
| `--integral_clip` | 1.0 | Symmetric clip on the PID integral |
| `--combine` | `multiply` | `multiply` -> use both alpha and `w(i)`; `replace` -> alpha replaces `w(i)` |

`make_controller('original')` returns `None`, so `--mode original`
short-circuits to the original sampler without constructing controller
state.

## Running

### Original AdaEdit (unchanged entry point)

```bash
python adaedit.py \
    -i examples/cat.jpg \
    -sp "A photo of a cat on a sofa" \
    -tp "A photo of a dog on a sofa" \
    --edit_object "cat" \
    --inject_schedule sigmoid \
    --use_channel_ls \
    --seed 42
```

### Adaptive variant (new entry point)

```bash
python adaedit_adaptive.py \
    -i examples/cat.jpg \
    -sp "A photo of a cat on a sofa" \
    -tp "A photo of a dog on a sofa" \
    --edit_object "cat" \
    --mode pd_adaptive \
    --drift_metric latent_init \
    --combine multiply \
    --kp 1.0 --kd 0.2 \
    --target_drift 0.05 \
    --base_alpha 1.0 --alpha_min 0.2 --alpha_max 1.2 \
    --seed 42 \
    --output_dir outputs_adaptive
```

### Original through the adaptive runner (for strict comparison)

```bash
python adaedit_adaptive.py \
    -i examples/cat.jpg -sp "..." -tp "..." --edit_object "cat" \
    --mode original --seed 42 \
    --output_dir outputs_adaptive
```

### PID variant

```bash
python adaedit_adaptive.py \
    -i examples/cat.jpg -sp "..." -tp "..." --edit_object "cat" \
    --mode pid_adaptive --kp 1.0 --ki 0.05 --kd 0.2 \
    --target_drift 0.05 --integral_clip 1.0 \
    --seed 42 --output_dir outputs_adaptive
```

### Run with metrics

Add `--metrics`. Produces `metrics.json` in the run directory:

```bash
python adaedit_adaptive.py ... --metrics
```

### Run without metrics (fast)

Omit `--metrics`. Only `edited.jpg` and `adaptive_log.json` are
written.

### Disable per-step logging

```bash
python adaedit_adaptive.py ... --no_log_controller
```

## Output layout

```
outputs_adaptive/
└── pd_adaptive_sigmoid_inject4/
    ├── edited.jpg            # decoded edited image
    ├── adaptive_log.json     # per-step drift, alpha, kv_mix_ratio, ...
    └── metrics.json          # only when --metrics
```

Repeated runs with the same name get a `_NNN` suffix rather than
overwriting.

`adaptive_log.json` contains:
- top-level run config (mode, gains, seed, schedule, ...)
- `per_step`: list of `{step, t_curr, t_prev, inject_weight, drift,
  alpha, kv_mix_ratio, base_kv_mix_ratio}`
- `controller_history`: structured `ControllerStep` entries
  (proportional/integral/derivative breakdown)

This is enough to plot the controller trajectory and compare against
the fixed-schedule baseline.

## Metrics

Intentionally lightweight (numpy + PIL only) so the runner works
without extra installs. Reported in `metrics.json`:

- `masked_mse_bg` — MSE on background / preservation region.
- `masked_psnr_bg` — PSNR on background (main preservation number).
- `ssim_bg` — coarse SSIM on background.
- `mean_abs_edit` — mean |src - out| inside edit region (does
  something change where we asked it to).

The edit mask is derived from `info['indices']` after Phase-1 mask
extraction and upsampled with nearest-neighbour to source resolution.

## Ablation matrix (suggested)

| Mode | `w(i)` | Controller | Expected role |
|------|--------|-----------|---------------|
| `original` | sigmoid | none | baseline |
| `fixed_soft` | sigmoid | const | isolates "constant soft" vs feedback |
| `pd_adaptive` | sigmoid | PD | main method |
| `pid_adaptive` | sigmoid | PID | extra stability test |

Run each with the same `--seed` and compare `metrics.json`.

## Notes / caveats

- Only the **target pass** is controlled. The inversion pass is
  identical to vanilla AdaEdit, so feature caching and mask extraction
  are byte-for-byte preserved.
- The controller resets at the start of every target pass.
- For `edit_type = style`, Phase 2 is skipped in the original
  pipeline; the adaptive runner follows the same convention.
- The drift metric operates on packed latents (`[B, L, C]`). Attention
  / feature-space drift would require hooking a layer — deliberately
  out of scope to keep the change minimal.
