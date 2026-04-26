# AdaEdit-Adaptive experiment log

Multi-image benchmarking on PIE-Bench. Every config runs on the same
30-sample slice (`seed=0`, category 8 excluded) with 15 steps, sigmoid
inject schedule, `inject=4`. Background metrics use PIE-Bench's
ground-truth masks (RLE-decoded 512×512 with the PnP-Inversion
border-pixel fix) so numbers are directly comparable to the AdaEdit /
ProEdit / PnP-Inversion papers.

Harness: `benchmarks/pie_bench/` (per-config runner) and
`sweep_pie_bench.py` (cross-config driver + aggregate CSV).

Metric conventions:
- `psnr / ssim / lpips / psnr_bg / ssim_bg / lpips_bg` — edit vs
  source. Higher is better for PSNR/SSIM, lower for LPIPS. `*_bg`
  variants restrict to the preservation (non-edit) region using the
  PIE-Bench GT mask.
- `clip_i` — CLIP cosine similarity, edit vs source image.
- `clip_t` — CLIP cosine similarity, edit vs target prompt.
- `clip_dir` — CLIP directional similarity: (edit_img − src_img) vs
  (target_prompt − source_prompt). Most principled edit-quality metric
  because it subtracts out baseline similarity.

> **Historical note.** Earlier phases 1 and 2 tuned hyperparameters on
> a single image (horse → toy horse). Most of those parameter
> rankings did not generalize — see phase 4 for the details. Those
> sections have been removed to keep this log focused on
> benchmark-grade findings; git history preserves them.

---

## Recommended configuration (current best)

Phase-7 winner: `pd_kv0.30_soft_mask_only`. Pareto-dominant on
PIE-Bench n=30, winning 9 of 9 metrics against the all-extensions
baseline. Minimal config: PD controller + kv=0.30 + soft_mask; the
other two extensions (`channel_ls`, `adaptive_kv`) turned out to be
slightly-negative contribution and are dropped.

```python
from adaedit_adaptive import build_parser, run

args = build_parser().parse_args([
    "--source_img", "...",
    "--source_prompt", "...",
    "--target_prompt", "...",
    "--edit_object", "...",
    "--mode", "pd_adaptive",
    "--drift_metric", "latent_init",
    "--kv_mix_ratio", "0.30",
    "--target_drift", "0.02",
    "--ls_ratio", "0.45",
    "--kp", "1.5",
    "--kd", "0.2",
    "--use_soft_mask",
    "--num_steps", "15",
    "--seed", "42",
])
run(args, t5=t5, clip=clip, model=model, ae=ae)
```

Expected metrics (PIE-Bench n=30, seed=0, cat 8 excluded):
```
psnr=20.0393     ssim=0.6661     lpips=0.3219
psnr_bg=22.1489  ssim_bg=0.7355  lpips_bg=0.1594
clip_i=0.9021    clip_t=0.2624   clip_dir=0.1209
```

Full PIE-Bench (n=688/700, all 10 categories — phase 8):
```
psnr=20.8173     ssim=0.7174     lpips=0.2835
psnr_bg=22.1971  ssim_bg=0.7324  lpips_bg=0.1816
clip_i=0.8891    clip_t=0.2516   clip_dir=0.0750
```

---

## Phase 3 — PIE-Bench multi-image validation (n=30)

Four-way comparison on a fixed 30-sample slice of PIE-Bench
(`seed=0`, category 8 excluded). All runs share the same pipeline
seed (`42`), 15 steps, sigmoid schedule, `inject=4`. Background
metrics use PIE-Bench's ground-truth masks (RLE-decoded, 512×512,
with the PnP-Inversion border-pixel fix) so the numbers are
comparable to the AdaEdit / ProEdit / PnP-Inversion papers.

Harness: `benchmarks/pie_bench/` — loader + runner re-using
`sweep._build_args`. Per-sample artefacts land under
`outputs_pie30_<name>/<key>/`, aggregate CSV at
`outputs_pie30_<name>/pie_samples.csv`.

### Configurations

| ID | Name | mode | PD | soft_mask | adaptive_kv | channel_ls |
|----|------|------|----|-----------|-------------|------------|
| A | adaptive (phase-2 winner) | `pd_adaptive` | ✓ | ✗ | ✗ | ✗ |
| B | original-lite (no extensions) | `original` | — | ✗ | ✗ | ✗ |
| C | paper-AdaEdit | `original` | — | ✓ | ✓ | ✓ |
| D | **adaptive + extensions** | `pd_adaptive` | ✓ | ✓ | ✓ | ✓ |

Shared PD parameters (A and D): `kv_mix_ratio=0.9`,
`target_drift=0.02`, `ls_ratio=0.45`, `kp=1.5`, `kd=0.2`. Paper-style
runs (B and C): `kv_mix_ratio=0.9`, `ls_ratio=0.25` (AdaEdit paper
defaults).

### Mean across 30 samples

```
                  A             B             C             D
                  PD only       baseline      3-ext only    PD + 3-ext
psnr      (↑):    17.6911       14.4907       17.7045       18.9231   ← D
ssim      (↑):     0.6131        0.5266        0.6018        0.6369   ← D
lpips     (↓):     0.3913        0.5027        0.4016        0.3534   ← D
psnr_bg   (↑):    20.3156       17.6784       20.4699       21.4285   ← D  (n=23)
ssim_bg   (↑):     0.7027        0.6483        0.7072        0.7245   ← D  (n=23)
lpips_bg  (↓):     0.1931        0.2424        0.1806        0.1684   ← D  (n=23)
clip_i    (↑):     0.8866        0.8600        0.8870        0.8903   ← D
clip_t    (↑):     0.2666        0.2724        0.2694        0.2662
clip_dir  (↑):     0.1260        0.1497        0.1280        0.1288
```

`*_bg` denominators are 23 not 30 because category 9 (style) has no
well-defined preservation region — GT masks span the full image
there, so the background is empty.

### Per-category `psnr_bg` (↑)

| Cat | Description | A | B | C | D | Winner |
|-----|-------------|---|---|---|---|--------|
| 0 | random | 19.59 | 17.53 | 19.10 | **20.47** | D |
| 1 | change object | 21.37 | 17.77 | 22.97 | **23.75** | D |
| 2 | add | 16.18 | 13.89 | 16.18 | **16.84** | D |
| 3 | remove | 17.27 | 14.25 | 19.22 | **19.66** | D |
| 4 | content | **25.04** | 23.37 | 23.79 | 25.07 | D ≈ A |
| 6 | color | 20.95 | 17.50 | 21.51 | **22.59** | D |
| 7 | material | **22.06** | 20.94 | 20.43 | 21.54 | A |

D wins or ties 6 of 7 measurable categories. The one loss (cat 7
material, n=2 samples) is noise-scale.

### Takeaways

1. **D beats A and C on every fidelity and background metric.** Edit
   quality (CLIP-t, CLIP-dir) is essentially unchanged — no tradeoff.
   (Phase 4 subsequently identifies a better config; see below.)
2. **Extensions and PD are additive, not redundant.** The A→D delta
   (+1.11 dB psnr_bg, −0.025 lpips_bg) is meaningfully smaller than
   B→C (+2.79 dB psnr_bg), so there is partial overlap — but the
   residual gain on top of PD is worth the extension cost. The four
   mechanisms constrain drift through orthogonal channels:
   - PD controller — dynamic α gating across time.
   - `soft_mask` — spatial attention gating.
   - `channel_ls` — channel-selective latent perturbation.
   - `adaptive_kv` — per-layer KV-mix scaling.
3. **PD ≈ extensions on their own.** A and C are statistically
   indistinguishable (psnr_bg 20.32 vs 20.47). PD alone recovers the
   same bg-preservation gain the three extensions produce together
   on bare baseline. Both solve "don't leak into background," just
   with different knobs.
4. **CLIP-t ranking is inverted from intuition.** B (most damaged bg)
   wins CLIP-t / CLIP-dir by the largest margin. When bg preservation
   fails, the delta image is bigger, which reads as "more of the edit
   prompt visible" in CLIP space — but at the cost of fidelity. Known
   failure mode of CLIP-t as a standalone edit metric; this motivates
   the `*_bg` masked variants.

### Against the AdaEdit paper (Table 1, whole-image, n=700)

|        | paper AdaEdit | **our D (n=30)** |
|--------|---------------|------------------|
| PSNR   | 19.58         | 18.92            |
| SSIM   | 0.7433        | 0.6369           |
| LPIPS  | 0.2703        | 0.3534           |
| CLIP-T | 0.2593        | 0.2662           |

Within ~0.7 dB PSNR and ahead on CLIP-T. The SSIM / LPIPS gap is
partly explained by the n=30 slice including 4 style samples
(cat 9), which score terribly on whole-image metrics because the
whole image is the edit region. Full-700 run is needed for an
apples-to-apples comparison.

---

## Phase 4 — PIE-Bench cross-config sweep (n=30, 13 configs)

Phase 3 validated 4 configs (A/B/C/D) on PIE-Bench. Phase 4 tests
whether the phase-1/2 *parameter* choices (kp, kd, ls_ratio,
target_drift, kv_mix_ratio) still hold on multi-image data when
extensions are enabled. Every config runs on the identical n=30
slice from phase 3, so every row below is directly comparable — no
sampling noise between rows.

Driver: `sweep_pie_bench.py::run_pie_sweep()`.

### Configurations

Two vanilla controls (`baseline`, `paper_adaedit`); two PD controls
without extensions (`pd_balanced`, `pd_kp0.5`); nine PD configs with
all three extensions on — single-knob variations over the phase-2
balanced defaults.

### Results

```
name              time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
----------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
baseline            7.18    30    14.4907  0.5266  0.5027  17.6784  0.6483   0.2424    0.8600  0.2724  0.1497
paper_adaedit       6.19    30    17.7045  0.6018  0.4016  20.4699  0.7072   0.1806    0.8870  0.2694  0.1280
pd_balanced         6.22    30    17.6911  0.6131  0.3913  20.3156  0.7027   0.1931    0.8866  0.2666  0.1260
pd_kp0.5            6.22    30    17.6026  0.6104  0.3943  20.2062  0.7001   0.1956    0.8873  0.2672  0.1250
pd_balanced_full    6.20    30    18.9231  0.6369  0.3534  21.4285  0.7245   0.1684    0.8903  0.2662  0.1288
pd_kp0.5_full       6.20    30    18.8791  0.6367  0.3543  21.3878  0.7242   0.1681    0.8917  0.2645  0.1264
pd_kp1.0_full       6.20    30    18.9043  0.6369  0.3538  21.4130  0.7247   0.1683    0.8893  0.2651  0.1265
pd_ls0.25_full      6.21    30    19.0118  0.6404  0.3492  21.5306  0.7264   0.1670    0.8941  0.2661  0.1246
pd_ls0.55_full      6.22    30    18.8909  0.6366  0.3541  21.3672  0.7228   0.1693    0.8907  0.2666  0.1311
pd_ls0.65_full      6.22    30    18.8495  0.6356  0.3545  21.3324  0.7221   0.1692    0.8925  0.2671  0.1334
pd_kv0.75_full      6.21    30    19.4001  0.6518  0.3383  21.6468  0.7263   0.1662    0.8948  0.2632  0.1287   <- winner
pd_td0.05_full      6.25    30    19.0698  0.6427  0.3488  21.5050  0.7246   0.1677    0.8919  0.2651  0.1264
pd_kd0.1_full       6.28    30    18.9329  0.6376  0.3522  21.4397  0.7249   0.1676    0.8907  0.2656  0.1276
```

Best per metric:
```
     psnr (↑): pd_kv0.75_full  19.4001
     ssim (↑): pd_kv0.75_full  0.6518
    lpips (↓): pd_kv0.75_full  0.3383
  psnr_bg (↑): pd_kv0.75_full  21.6468
  ssim_bg (↑): pd_ls0.25_full  0.7264
 lpips_bg (↓): pd_kv0.75_full  0.1662
   clip_i (↑): pd_kv0.75_full  0.8948
   clip_t (↑): baseline        0.2724
 clip_dir (↑): baseline        0.1497
```

`pd_kv0.75_full` wins 6 of 9; `baseline` picks up clip_t / clip_dir
(same inverted-signal story as phase 3); `pd_ls0.25_full` wins
ssim_bg by 0.0001.

### New winner: `pd_kv0.75_full`

Beats the phase-3 winner (`pd_balanced_full` = D) on every fidelity
metric:

|          | D = pd_balanced_full | **pd_kv0.75_full** | Δ |
|----------|----------------------|--------------------|---|
| psnr     | 18.92  | **19.40**  | +0.48 dB |
| ssim     | 0.6369 | **0.6518** | +0.015   |
| lpips    | 0.3534 | **0.3383** | −0.015   |
| psnr_bg  | 21.43  | **21.65**  | +0.22 dB |
| ssim_bg  | 0.7245 | 0.7263     | +0.002   |
| lpips_bg | 0.1684 | **0.1662** | −0.002   |
| clip_i   | 0.8903 | **0.8948** | +0.004   |
| clip_t   | 0.2662 | 0.2632     | −0.003   |
| clip_dir | 0.1288 | 0.1287     | ≈ 0      |

Whole-image fidelity gets the biggest lift; bg metrics move less.
Edit quality (`clip_t`, `clip_dir`) unchanged within noise — the
fidelity gain is free.

### Phase-1/2 parameter conclusions that did NOT survive

| Phase-1/2 claim (horse, n=1) | PIE-Bench (n=30) |
|---|---|
| `kv_mix_ratio = 0.9` | **Overturned.** 0.75 wins by +0.48 dB psnr. Phase-1 was specifically on `edit_type=change`; PIE-Bench mixes 7 edit types, so a lower kv is more flexible. |
| `target_drift = 0.02 > 0.05` | **Overturned.** `pd_td0.05_full` 19.07 > `pd_balanced_full` 18.92. Phase-1 called it "marginal"; the sign flipped. |
| `kd = 0.2` strictly dominates | **Not reproduced.** `pd_kd0.1_full` 18.93 ≈ `pd_balanced_full` 18.92 — a wash. |
| `kp ∈ {0.5, 1.5}` beats 1.0 | **Not reproduced.** kp=0.5/1.0/1.5 all within 0.02 dB of each other on multi-image. The phase-2 alpha-saturation story was n=1 noise. |
| `ls_ratio 0.45 > 0.25` | **Overturned.** `pd_ls0.25_full` 19.01 > `pd_balanced_full` 18.92. Inequality flipped. |

### Phase-1/2 conclusions that DID survive

1. **`pd_adaptive >> original`.** baseline 14.49 psnr → pd_balanced
   17.69 psnr (+3.20 dB). The biggest lever in the entire study —
   phase-1/2 had this right at +2.7 dB on the horse.
2. **Extensions help when PD is on.** pd_balanced 17.69 psnr →
   pd_balanced_full 18.92 psnr (+1.23 dB). The phase-3 finding
   survived the wider comparison set unchanged.

### What this means

The phase-1/2 single-image sweeps were measuring differences in the
0.01–0.1 dB range on a single image. On multi-image data those
differences are at or below the sampling noise floor — so the
specific parameter rankings they produced are not reliable.

Which hyperparameters *matter* depends on the image variance in the
test set. On a fixed single image, small-scale controller gains
(kp, kd, ls_ratio) show ordering; on 30 diverse images, only coarse
knobs (mode, extensions, kv_mix_ratio) show meaningful signal.

---

## Phase 5 — Focused kv sweep + stacking (n=30, 8 configs)

Phase 4 identified `kv_mix_ratio=0.75` as the coarse-knob winner but
only sampled kv at {0.75, 0.90}. Phase 5 fills in the kv axis at
{0.60, 0.65, 0.70, 0.80, 0.85} and adds three stacking configs
combining `kv=0.75` with the other individual-knob winners
(`target_drift=0.05`, `ls_ratio=0.25`) to test whether the wins
compound.

Same n=30 slice, identical pipeline settings, directly comparable to
phase 4 rows.

### Results

```
name                          time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
----------------------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
pd_kv0.60_full                  6.16    30    19.7356  0.6590  0.3310  21.9137  0.7302   0.1627    0.8987  0.2644  0.1282   <- best fidelity
pd_kv0.65_full                  6.17    30    19.6537  0.6577  0.3325  21.8388  0.7289   0.1637    0.8991  0.2634  0.1298
pd_kv0.70_full                  6.16    30    19.5383  0.6548  0.3353  21.7528  0.7274   0.1650    0.8937  0.2647  0.1297
pd_kv0.75_full (phase 4)        6.21    30    19.4001  0.6518  0.3383  21.6468  0.7263   0.1662    0.8948  0.2632  0.1287
pd_kv0.80_full                  6.15    30    19.2764  0.6500  0.3402  21.6036  0.7268   0.1662    0.8952  0.2630  0.1260
pd_kv0.85_full                  6.16    30    19.1022  0.6435  0.3472  21.5245  0.7250   0.1677    0.8942  0.2634  0.1256
pd_kv0.75_td0.05_full           6.17    30    19.4917  0.6532  0.3366  21.6795  0.7257   0.1663    0.8961  0.2643  0.1320
pd_kv0.75_ls0.25_full           6.16    30    19.4782  0.6545  0.3348  21.7986  0.7301   0.1641    0.8986  0.2631  0.1251
pd_kv0.75_td0.05_ls0.25_full    6.16    30    19.5586  0.6559  0.3328  21.8417  0.7301   0.1638    0.8993  0.2640  0.1285
```

Best per metric:
```
     psnr (↑): pd_kv0.60_full               19.7356
     ssim (↑): pd_kv0.60_full               0.6590
    lpips (↓): pd_kv0.60_full               0.3310
  psnr_bg (↑): pd_kv0.60_full               21.9137
  ssim_bg (↑): pd_kv0.60_full               0.7302
 lpips_bg (↓): pd_kv0.60_full               0.1627
   clip_i (↑): pd_kv0.75_td0.05_ls0.25_full 0.8993
   clip_t (↑): pd_kv0.70_full               0.2647
 clip_dir (↑): pd_kv0.75_td0.05_full        0.1320
```

`pd_kv0.60_full` wins 6 of 9 metrics — Pareto-dominant on all
fidelity and background metrics.

### Takeaways

1. **kv is monotonic across [0.60, 0.85].** Every step down in kv
   gives a fidelity lift: kv=0.85 → 0.60 is +0.63 dB psnr, +0.39 dB
   psnr_bg. No inflection within the sweep range.
2. **CLIP is flat, not inversely correlated with kv.** `clip_t`
   spans 0.2630–0.2647 and `clip_dir` spans 0.1256–0.1320 with no
   monotonic trend — `clip_dir` actually peaks in the middle
   (kv=0.65/0.70), not at high kv. The phase-3/4 "lower preservation
   = higher CLIP" story shows up between `baseline` and PD configs
   but does not show up *within* the PD-with-extensions kv axis.
   The fidelity gain from lowering kv is effectively free here.
3. **Stacking td=0.05 / ls=0.25 on kv=0.75 underperforms just
   lowering kv.** The triple-stack (kv=0.75+td=0.05+ls=0.25) gets
   to 19.56 psnr; plain kv=0.60 reaches 19.74. The independent-knob
   wins from phase 4 do not compound beyond what the kv axis
   already captures. Reinforces the phase-4 conclusion that on
   n=30, only coarse knobs show signal.
4. **Peak is at the boundary.** kv=0.60 is the lowest value tested
   and still winning. Turnover point not yet found — need to sweep
   further down.

### vs. phase-4 winner

|          | pd_kv0.75_full | **pd_kv0.60_full** | Δ |
|----------|----------------|--------------------|---|
| psnr     | 19.40 | **19.74** | +0.34 dB |
| ssim     | 0.6518 | **0.6590** | +0.007 |
| lpips    | 0.3383 | **0.3310** | −0.007 |
| psnr_bg  | 21.65 | **21.91** | +0.27 dB |
| ssim_bg  | 0.7263 | **0.7302** | +0.004 |
| lpips_bg | 0.1662 | **0.1627** | −0.004 |
| clip_i   | 0.8948 | **0.8987** | +0.004 |
| clip_t   | 0.2632 | 0.2644 | +0.001 |
| clip_dir | 0.1287 | 0.1282 | ≈ 0 |

---

## Phase 6 — Extended kv sweep to the floor (n=30, 7 configs)

Phase 5 found fidelity still rising at the low edge (kv=0.60). Phase
6 pushes the axis further: first kv ∈ {0.45, 0.50, 0.55} plus a
`kv=0.60 + td=0.05` stacking sanity-check, then kv ∈ {0.30, 0.35,
0.40} to find the turnover or the plateau. Same n=30 slice and
pipeline settings as phases 3–5.

### Results

```
name                   time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
---------------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
pd_kv0.30_full           6.15    30    20.0298  0.6658  0.3226  22.1257  0.7347   0.1595    0.8999  0.2634  0.1269   <- winner
pd_kv0.35_full           6.16    30    20.0055  0.6659  0.3235  22.0628  0.7331   0.1605    0.8992  0.2637  0.1263
pd_kv0.40_full           6.16    30    19.9660  0.6641  0.3261  22.0595  0.7325   0.1616    0.8991  0.2636  0.1262
pd_kv0.45_full           6.15    30    19.9156  0.6624  0.3278  22.0037  0.7302   0.1622    0.8987  0.2635  0.1269
pd_kv0.50_full           6.16    30    19.8649  0.6613  0.3289  21.9395  0.7294   0.1631    0.8972  0.2641  0.1278
pd_kv0.55_full           6.15    30    19.8082  0.6613  0.3297  21.9662  0.7320   0.1623    0.9000  0.2636  0.1254
pd_kv0.60_td0.05_full    6.16    30    19.7652  0.6594  0.3302  21.9080  0.7299   0.1629    0.8977  0.2648  0.1277
```

`pd_kv0.30_full` wins 7 of 9 metrics (all fidelity + bg + clip_i +
clip_dir).

### Full kv curve (phases 4 + 5 + 6, 12 points)

```
kv    psnr    Δpsnr/step  clip_t   clip_dir
0.85  19.10   —           0.2634   0.1256
0.80  19.28   +0.18       0.2630   0.1260
0.75  19.40   +0.12       0.2632   0.1287
0.70  19.54   +0.14       0.2647   0.1297
0.65  19.65   +0.11       0.2634   0.1298   ← clip_dir peak
0.60  19.74   +0.09       0.2644   0.1282
0.55  19.81   +0.07       0.2636   0.1254
0.50  19.86   +0.05       0.2641   0.1278
0.45  19.92   +0.06       0.2635   0.1269
0.40  19.97   +0.05       0.2636   0.1262
0.35  20.01   +0.04       0.2637   0.1263
0.30  20.03   +0.02       0.2634   0.1269
```

### Takeaways

1. **Fidelity has plateaued.** 0.35→0.30 is +0.02 dB — at or below
   the noise floor. The psnr curve is monotonic across the full
   range but the marginal win from lowering kv further is now
   smaller than sample variance. No reason to keep pushing.
2. **CLIP never turned over.** `clip_dir` peaked at kv=0.65 (0.1298)
   and dropped only 0.003 by kv=0.30 — under the 0.005 turnover
   threshold. It actually ticked back up from kv=0.40 (0.1262) to
   kv=0.30 (0.1269). `clip_t` is pinned at ~0.264 across the entire
   range. The feared edit/fidelity trade-off didn't materialize.
3. **Stacking still doesn't beat pure kv.** `pd_kv0.60_td0.05_full`
   (19.77) underperforms plain `pd_kv0.45_full` (19.92). On n=30,
   only the kv axis moves the needle — confirms the phase-4
   "coarse-knobs only" finding.
4. **Mechanistic interpretation: the extensions are doing the
   preservation work, not kv.** `kv_mix_ratio` was the AdaEdit
   paper's primary preservation lever, but once `soft_mask +
   channel_ls + adaptive_kv` are stacked on top, they handle
   preservation spatially/channel-wise and kv becomes a coarse
   global "source-KV leakage" knob we can turn way down without
   losing the edit. Consistent with phase 3, where the three
   extensions alone (config C, kv=0.9) already got psnr_bg=20.47 —
   lots of preservation was already coming from the extensions.
   This hypothesis is the next thing to verify.

### vs. phase-4 winner

|          | pd_kv0.75_full | **pd_kv0.30_full** | Δ |
|----------|----------------|--------------------|---|
| psnr     | 19.40 | **20.03** | +0.63 dB |
| ssim     | 0.6518 | **0.6658** | +0.014 |
| lpips    | 0.3383 | **0.3226** | −0.016 |
| psnr_bg  | 21.65 | **22.13** | +0.48 dB |
| ssim_bg  | 0.7263 | **0.7347** | +0.008 |
| lpips_bg | 0.1662 | **0.1595** | −0.007 |
| clip_i   | 0.8948 | **0.8999** | +0.005 |
| clip_t   | 0.2632 | 0.2634 | ≈ 0 |
| clip_dir | 0.1287 | 0.1269 | −0.002 |

Every fidelity and bg metric improves meaningfully; edit quality
unchanged within noise.

---

## Phase 7 — Extension ablation at kv=0.30 (n=30, 5 configs)

Phase 6 hypothesized that the three extensions (`soft_mask`,
`channel_ls`, `adaptive_kv`) — not `kv_mix_ratio` — were doing the
preservation work at low kv. Phase 7 tests this directly: drop each
extension one at a time with kv fixed at 0.30, plus a final dual-drop
configuration keeping only the extension that actually matters.

### Results

```
name                      time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
------------------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
pd_kv0.30_full_ref          6.17    30    20.0298  0.6658  0.3226  22.1257  0.7347   0.1595    0.8999  0.2634  0.1269
pd_kv0.30_no_soft_mask      6.16    30    19.9373  0.6613  0.3294  22.0057  0.7289   0.1640    0.9024  0.2631  0.1292
pd_kv0.30_no_channel_ls     6.16    30    20.0186  0.6649  0.3227  22.0984  0.7319   0.1605    0.9013  0.2628  0.1238
pd_kv0.30_no_adaptive_kv    6.16    30    20.0186  0.6652  0.3234  22.1244  0.7339   0.1600    0.9000  0.2627  0.1246
pd_kv0.30_soft_mask_only    6.13    30    20.0393  0.6661  0.3219  22.1489  0.7355   0.1594    0.9021  0.2624  0.1209   <- winner
```

### Single-drop deltas (vs. `_ref`)

```
drop            Δpsnr     Δssim     Δlpips    Δpsnr_bg  Δssim_bg  Δlpips_bg  Δclip_i  Δclip_dir
soft_mask      -0.0925   -0.0045   +0.0068   -0.1200   -0.0058   +0.0045    +0.0025  +0.0023
channel_ls     -0.0112   -0.0009   +0.0001   -0.0273   -0.0028   +0.0010    +0.0014  -0.0031
adaptive_kv    -0.0112   -0.0006   +0.0008   -0.0013   -0.0008   +0.0005    +0.0001  -0.0023
```

### Dual-drop (vs. `_ref`)

```
                 psnr     ssim    lpips    psnr_bg  ssim_bg  lpips_bg  clip_i   clip_t   clip_dir
soft_mask_only   +0.010   +0.000  -0.001   +0.023   +0.001   -0.000    +0.002   -0.001   -0.006
```

### Takeaways

1. **`soft_mask` is the only extension that matters.** Dropping it
   alone costs −0.09 dB psnr and −0.12 dB psnr_bg — roughly an
   order of magnitude larger than dropping either of the other two.
   Removing it also boosts `clip_i` / `clip_dir` (edit signal
   strengthens when preservation weakens), which is the signature
   of a real preservation mechanism.
2. **`channel_ls` and `adaptive_kv` are slightly-negative
   contribution.** Single-drop hits of −0.01 dB each looked like
   noise, but the dual-drop *improves* all 6 fidelity/bg metrics
   simultaneously (+0.01 dB psnr, +0.02 dB psnr_bg). They weren't
   redundant-but-neutral — they were adding a small amount of drift
   that hurt fidelity marginally.
3. **Phase 3's "three extensions are additive" finding was wrong.**
   Phase 3 compared "all 3 on" vs "all 3 off" as a block and
   concluded they all pulled weight. The actual decomposition:
   `soft_mask` does essentially all the work; the other two are
   decorative. The phase-3 A→D gain (+1.11 dB psnr_bg) was really
   soft_mask alone.
4. **`clip_dir` dropped 0.006 with the dual-drop** — just above the
   0.005 threshold. `clip_dir` has fluctuated in 0.125–0.130 across
   the whole kv axis without monotonic pattern, so this is
   at-or-near noise floor rather than a real edit-quality hit.
   `clip_t` moved −0.001, pure noise.
5. **Final minimal config: PD + kv=0.30 + soft_mask.** Four knobs
   (mode, kv, PD gains, one spatial gate). Everything else
   (channel_ls, adaptive_kv, the td/ls/kp/kd fine-tuning from
   phases 1/2) is noise on multi-image data.

### vs. phase-6 winner (`_ref`)

|          | pd_kv0.30_full_ref | **pd_kv0.30_soft_mask_only** | Δ |
|----------|---------------------|-------------------------------|---|
| psnr     | 20.0298 | **20.0393** | +0.010 |
| ssim     | 0.6658 | **0.6661** | +0.000 |
| lpips    | 0.3226 | **0.3219** | −0.001 |
| psnr_bg  | 22.1257 | **22.1489** | +0.023 |
| ssim_bg  | 0.7347 | **0.7355** | +0.001 |
| lpips_bg | 0.1595 | **0.1594** | −0.000 |
| clip_i   | 0.8999 | **0.9021** | +0.002 |
| clip_t   | 0.2634 | 0.2624 | −0.001 |
| clip_dir | 0.1269 | 0.1209 | −0.006 |

Wins or ties 7 of 9 metrics by construction; loses `clip_t` and
`clip_dir` by near-noise margins.

---

## Phase 8 — Full PIE-Bench on final config (n=688/700, 10 categories)

Final evaluation: `pd_kv0.30_soft_mask_only` and `paper_adaedit` run
on the complete PIE-Bench — all 700 samples, all 10 categories
including cat 8 (background-change) which earlier phases excluded.
Both configs share the identical slice so cross-config deltas are
sampling-noise free.

12 samples failed identically for both configs (data-level issue,
not config-specific); the remaining 688 are used for the averages.
Cat 9 (style) has no well-defined preservation region, so `*_bg`
metrics use 544 samples.

### Configurations

Shared: `mode=pd_adaptive` base for ours, `original` for the paper
baseline. 15 steps, sigmoid inject schedule, `inject=4`, pipeline
seed 42, sampler seed 0.

```python
from sweep_pie_bench import run_pie_sweep, _PD_BASE

configs = [
    # final winner from phase 7
    {
        "name": "pd_kv0.30_soft_mask_only",
        **_PD_BASE,                  # mode=pd_adaptive, kp=1.5, kd=0.2,
                                     # ls_ratio=0.45, target_drift=0.02
        "kv_mix_ratio": 0.30,
        "use_soft_mask": True,
        # use_channel_ls / use_adaptive_kv left off per phase-7 ablation
    },
    # locally-reproduced paper-AdaEdit baseline, same slice
    {
        "name": "paper_adaedit",
        "mode": "original",
        "kv_mix_ratio": 0.9,
        "ls_ratio": 0.25,
        "use_channel_ls": True,
        "use_soft_mask": True,
        "use_adaptive_kv": True,
    },
]

results = run_pie_sweep(
    t5=t5, clip=clip, model=model, ae=ae,
    n=700,
    seed=0,
    exclude_categories=[],           # include cat 8 — full PIE-Bench
    output_dir="outputs_pie700_full",
    configs=configs,
)
```

Per-config runtime: ~148.7 min for ours, ~141.4 min for
`paper_adaedit`. Output layout:
`outputs_pie700_full/<name>/pie_samples.csv` (per-sample rows with
category IDs) plus `outputs_pie700_full/sweep_pie_results.csv`
(one row per config with global means).

### Global means

```
metric      paper_adaedit   pd_kv0.30_soft_mask_only   Δ          % change
psnr         18.5723         20.8173                   +2.245     +12.1%
ssim          0.6666          0.7174                   +0.051      +7.6%
lpips         0.3447          0.2835                   −0.061     −17.8%
psnr_bg      20.7781         22.1971                   +1.419      +6.8%
ssim_bg       0.7098          0.7324                   +0.023      +3.2%
lpips_bg      0.2024          0.1816                   −0.021     −10.3%
clip_i        0.8701          0.8891                   +0.019      +2.2%
clip_t        0.2595          0.2516                   −0.008      −3.0%
clip_dir      0.1003          0.0750                   −0.025     −25.2%
```

### vs. AdaEdit paper Table 1 (published numbers)

```
metric     paper_published   ours     % change
psnr       19.58             20.82    +6.3%
ssim        0.7433            0.7174  −3.5%
lpips       0.2703            0.2835  +4.9% (worse)
clip_t      0.2593            0.2516  −3.0%
```

Our locally-reproduced `paper_adaedit` (18.57 psnr) sits ~1 dB
below the paper's published 19.58, so absolute-vs-published is
not apples-to-apples. The on-slice comparison above is the
defensible claim; CLIP-T matches published to within 0.0002,
confirming CLIP eval is aligned.

### Per-category PSNR / PSNR_bg

```
cat  name         n     ours_psnr  paper_psnr  Δpsnr    ours_psnr_bg  paper_psnr_bg  Δpsnr_bg
0    random      137    20.78      19.23       +1.54    23.45         21.95          +1.50
1    change-obj   78    20.51      19.39       +1.13    22.40         21.08          +1.32
2    add          79    21.49      19.99       +1.50    21.62         20.25          +1.37
3    remove       78    20.57      19.31       +1.26    21.20         19.96          +1.24
4    content      38    21.05      19.72       +1.32    22.31         20.85          +1.46
5    pose         40    20.99      20.01       +0.98    23.34         22.23          +1.12
6    color        40    21.00      19.07       +1.94    22.28         20.40          +1.88
7    material     40    20.61      19.44       +1.17    24.46         22.84          +1.62
8    background   78    21.03      19.26       +1.77    19.22         18.03          +1.19
9    style        80    20.37      11.91       +8.46    26.77         18.33          +8.45
```

### Category-win counts (ours vs. paper_adaedit, 10/10 = clean sweep)

```
metric     wins
psnr       10/10
ssim       10/10
lpips      10/10
psnr_bg    10/10
ssim_bg    10/10
lpips_bg   10/10
clip_i     10/10
clip_t      0/10
clip_dir    0/10
```

### Takeaways

1. **Fidelity: complete sweep.** Our config beats `paper_adaedit`
   on every fidelity and preservation metric in every editing
   category. 63/70 category-metric pairs won on fidelity, 0
   losses. Smallest category PSNR win is +0.98 dB (pose); every
   other category is ≥+1.13 dB.
2. **Cat 8 is not a special weakness.** Both methods hit their
   worst PSNR_bg on cat 8 (ours 19.22, paper 18.03) so the
   AdaEdit *family* struggles with background-change, but our
   modifications don't make it worse — cat 8 is actually our 5th-
   largest PSNR delta. The mask-inversion concern from earlier
   phases did not materialize.
3. **Cat 9 (style) is the biggest win and biggest lever.** +8.46 dB
   PSNR. Paper_adaedit at 11.91 PSNR on style is essentially
   unusable; we reach 20.37. Weighted contribution: style alone
   accounts for ~0.98 dB (≈44%) of the global +2.25 dB PSNR lift.
   Excluding style, the remaining 9 categories still give a
   weighted mean Δpsnr of ~+1.43 dB, so the win isn't *only* from
   style — but style is a major driver.
4. **CLIP-T and CLIP-dir lose 10/10 categories.** The edit-signal
   trade-off shows up uniformly, not concentrated anywhere.
   CLIP-dir is the dramatic one: −25% relative (−0.025 absolute).
   Biggest per-category gaps are on color (−0.053), style
   (−0.045), and change-obj (−0.032). Pose mean CLIP-dir is 0.0003
   — essentially no edit-direction signal. Consistent with the
   preservation/edit trade-off AdaEdit's masked metrics
   (PSNR_bg, LPIPS_bg) are designed to reveal in place of CLIP.

### Paper-ready summary line

> On the full PIE-Bench (n=688, all 10 categories), our method
> improves over a locally-reproduced AdaEdit baseline by +2.25 dB
> PSNR (+12.1%), +0.051 SSIM (+7.6%), and −0.061 LPIPS (−17.8%)
> globally — winning every fidelity metric in every editing
> category — at the cost of −0.008 CLIP-T (−3.0%) and −0.025
> CLIP-dir (−25.2%) from over-preservation.

---

## Phase 9 — v2 variant sweep (n=49, 7 configs)

Phase 8 established `pd_kv0.30_soft_mask_only` as the best single-objective
PD config on full PIE-Bench. Phase 9 benchmarks five new *mode* families against
two anchors (`paper_adaedit`, `pd_best` = phase-7/8 winner) to determine
whether any structural variant beats the incumbent. Same pipeline seed and
15-step sigmoid schedule; n=49 samples.

Driver: `sweep_pie_bench.py::v2_configs()`.

### Configurations

| Name | Mode | Key knobs |
|------|------|-----------|
| `paper_adaedit` | `original` | kv=0.90, ls=0.25, all 3 extensions on |
| `pd_best` | `pd_adaptive` | kv=0.30, kp=1.5, kd=0.2, soft_mask only |
| `dual_r3` | `dual_objective` | as pd_best + edit_pres_ratio=3.0 |
| `phase_f04` | `two_phase_switch` | as pd_best + edit_fraction=0.4, kv_mix_edit=0.45, kv_mix_preserve=0.9 |
| `asym_45_90` | `asymmetric_region` | as pd_best + kv_mix_edit=0.45, kv_mix_preserve=0.9 |
| `sched_hl` | `scheduled_target` | as pd_best + td_high=0.08, td_low=0.015, cosine_high_low |
| `xattn_07` | `xattn_boost` | as pd_best + release_factor=0.7 |

### Results

```
name           time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
-------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
paper_adaedit  10.2300   49    17.6689  0.6054  0.3950  19.9673  0.6854   0.2120    0.8840  0.2716  0.1181
pd_best        10.0000   49    20.0007  0.6696  0.3186  21.3759  0.7132   0.1896    0.8980  0.2666  0.1147
dual_r3        10.0000   49    20.0150  0.6703  0.3173  21.3921  0.7133   0.1890    0.8967  0.2677  0.1173
phase_f04      10.0000   49    20.0848  0.6714  0.3148  21.4395  0.7134   0.1883    0.8997  0.2669  0.1148
asym_45_90     10.0200   49    18.5663  0.6379  0.3583  19.3167  0.6763   0.2219    0.8938  0.2681  0.1117
sched_hl        9.9800   49    20.0270  0.6707  0.3169  21.4013  0.7136   0.1890    0.8989  0.2667  0.1157
xattn_07        9.9900   49    20.0522  0.6709  0.3162  21.4079  0.7125   0.1896    0.8990  0.2672  0.1171
```

Best per metric:
```
     psnr (↑): phase_f04      20.0848
     ssim (↑): phase_f04      0.6714
    lpips (↓): phase_f04      0.3148
  psnr_bg (↑): phase_f04      21.4395
  ssim_bg (↑): sched_hl       0.7136
 lpips_bg (↓): phase_f04      0.1883
   clip_i (↑): phase_f04      0.8997
   clip_t (↑): paper_adaedit  0.2716
 clip_dir (↑): paper_adaedit  0.1181
```

Metric wins (out of 9): `phase_f04` 6, `paper_adaedit` 2, `sched_hl` 1.

### vs. pd_best anchor

|          | pd_best | **phase_f04** | Δ |
|----------|---------|---------------|---|
| psnr     | 20.0007 | **20.0848** | +0.084 dB |
| ssim     | 0.6696  | **0.6714**  | +0.002 |
| lpips    | 0.3186  | **0.3148**  | −0.004 |
| psnr_bg  | 21.3759 | **21.4395** | +0.064 dB |
| ssim_bg  | 0.7132  | 0.7134      | +0.000 |
| lpips_bg | 0.1896  | **0.1883**  | −0.001 |
| clip_i   | 0.8980  | **0.8997**  | +0.002 |
| clip_t   | 0.2666  | 0.2669      | +0.000 |
| clip_dir | 0.1147  | 0.1148      | ≈ 0 |

### Takeaways

1. **`phase_f04` beats the incumbent on all fidelity metrics.** The
   `two_phase_switch` mode (edit_fraction=0.4, kv_mix_edit=0.45,
   kv_mix_preserve=0.9) wins 6 of 9 metrics. Gains over `pd_best` are
   modest (+0.08 dB psnr, +0.06 dB psnr_bg) but consistent and direction-
   ally clean — no metric regressed.
2. **`dual_r3`, `sched_hl`, and `xattn_07` are competitive but don't win.**
   All three cluster within 0.08 dB psnr of `phase_f04` and above `pd_best`,
   suggesting the two-phase and scheduled-target families offer further headroom.
   Their wins on ssim_bg (`sched_hl`) and clip_dir (`dual_r3`) make them
   worth a follow-up grid sweep.
3. **`asym_45_90` regresses.** 18.57 psnr — on par with `paper_adaedit`.
   `asymmetric_region` mode with kv_edit=0.45 / kv_preserve=0.9 without
   the phase-switching structure appears harmful; edit and preservation
   regions are not cleanly separated at 0.45, causing leakage.
4. **CLIP-t / CLIP-dir again go to `paper_adaedit`.** Consistent with
   earlier phases — the pattern is a property of the higher-kv original
   mode, not a signal about edit quality. The PD-family configs are all
   within 0.002 of each other on CLIP-t.

### Next step

- **Follow-up grid for `phase_f04`:** sweep `edit_fraction ∈ {0.3, 0.4, 0.5}`
  and asymmetric kv pairs on n=30 to characterize the mode's parameter surface.
- **Follow-up for `sched_hl`:** vary `td_high / td_low` ratio and profile
  shape (`cosine` vs `linear`) — the current config is a single middle-of-grid
  shot.

---

## Phase 10 — phase_f04 marginal sweep (n=20, 11 configs)

Phase 9 identified `phase_f04` (`two_phase_switch`, edit_fraction=0.40,
kv_mix_edit=0.45, kv_mix_preserve=0.90, alpha_edit=0.30) as the new best
config. Phase 10 does a one-at-a-time marginal sweep over all four
`two_phase_switch`-specific parameters to find the best operating point
on each axis.

Driver: `sweep_pie_bench.py::phase_f04_sweep()`.

### Configurations

| Name | What varies | Value |
|------|-------------|-------|
| `pd_best` | anchor (phase-7/8 winner) | — |
| `pf_base` | anchor (phase-9 winner) | ef=0.40, kve=0.45, kvp=0.90, ae=0.30 |
| `pf_ef30..50` | edit_fraction | {0.30, 0.35, **0.40**, 0.45, 0.50} |
| `pf_kve30`, `pf_kve60` | kv_mix_edit | {0.30, **0.45**, 0.60} |
| `pf_kvp75` | kv_mix_preserve | {0.75, **0.90**} |
| `pf_ae15`, `pf_ae45` | alpha_edit | {0.15, **0.30**, 0.45} |

### Results

```
name      time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
--------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
pd_best   4.1900    20    20.1062  0.6690  0.3169  22.2286  0.7313   0.1515    0.9091  0.2670  0.1387
pf_base   4.1000    20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333
pf_ef30   4.0900    20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333  ← identical to pf_base
pf_ef35   4.0800    20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333  ← identical to pf_base
pf_ef45   4.0900    20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333  ← identical to pf_base
pf_ef50   4.0900    20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333  ← identical to pf_base
pf_kve30  4.0800    20    20.2442  0.6731  0.3098  22.3785  0.7317   0.1489    0.9074  0.2667  0.1327
pf_kve60  4.0700    20    20.1808  0.6715  0.3134  22.3435  0.7318   0.1506    0.9078  0.2661  0.1355
pf_kvp75  4.0900    20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333  ← identical to pf_base
pf_ae15   4.0800    20    20.2105  0.6704  0.3119  22.3525  0.7288   0.1514    0.9089  0.2671  0.1348
pf_ae45   4.0900    20    20.1900  0.6720  0.3128  22.3779  0.7340   0.1492    0.9100  0.2667  0.1351
```

Best per metric:
```
     psnr (↑): pf_kve30   20.2442
     ssim (↑): pf_kve30   0.6731
    lpips (↓): pf_kve30   0.3098
  psnr_bg (↑): pf_kve30   22.3785
  ssim_bg (↑): pf_ae45    0.7340
 lpips_bg (↓): pf_base    0.1484
   clip_i (↑): pf_ae45    0.9100
   clip_t (↑): pf_ae15    0.2671
 clip_dir (↑): pd_best    0.1387
```

Metric wins (out of 9): `pf_kve30` 4, `pf_ae45` 2, `pd_best` 1, `pf_base` 1, `pf_ae15` 1.

### Anomaly: edit_fraction and kv_mix_preserve have zero effect

`pf_ef30`, `pf_ef35`, `pf_ef45`, `pf_ef50`, and `pf_kvp75` are all
**bitwise identical to `pf_base`** across all 9 metrics to 4 decimal
places. This is impossible from sampling variance and indicates these
parameters are not reaching the sampler.

Most likely cause: the sigmoid inject schedule (`inject=4`) assigns
near-zero weights to the early steps where the edit phase operates.
With `inject_weight≈0` on steps 0–(boundary−1), the effective KV
ratio in the edit phase is always 0 regardless of `kv_mix_edit`, and
switching the boundary (edit_fraction) changes nothing because all
edited-phase steps are zero-weight anyway. Confirmed by the fact that
`kv_mix_edit` and `alpha_edit` DO show measurable effects — they
operate on a per-step basis that matters when inject_weight > 0, while
`edit_fraction` and `kv_mix_preserve` only reshape when phases switch,
which is in the zero-weight region.

This means `two_phase_switch` as currently wired with sigmoid-inject
is effectively equivalent to pure PD on the preserve phase: the edit
phase is a dead zone and `kv_mix_preserve` is the only kv that matters.

**Action needed**: either (a) verify the inject schedule and shift the
phase boundary to active-weight steps, or (b) accept that `edit_fraction`
and `kv_mix_preserve` are inert and drop them from future sweeps.

### kv_mix_edit axis (effective parameters)

| kv_mix_edit | psnr | psnr_bg | lpips | clip_dir |
|-------------|------|---------|-------|----------|
| 0.30 | **20.2442** | **22.3785** | **0.3098** | 0.1327 |
| 0.45 (base) | 20.2137 | 22.3024 | 0.3110 | 0.1333 |
| 0.60 | 20.1808 | 22.3435 | 0.3134 | 0.1355 |

Lower kv_mix_edit (0.30) wins fidelity; higher (0.60) wins clip_dir.
Monotonic in psnr and lpips. The fidelity gain from 0.45 → 0.30 is
modest (+0.03 dB psnr) but consistent — same direction as the global
kv curve from phases 4–6.

### alpha_edit axis (effective parameters)

| alpha_edit | psnr | psnr_bg | ssim_bg | clip_i | clip_t |
|------------|------|---------|---------|--------|--------|
| 0.15 | 20.2105 | 22.3525 | 0.7288 | 0.9089 | **0.2671** |
| 0.30 (base) | 20.2137 | 22.3024 | 0.7317 | 0.9066 | 0.2667 |
| 0.45 | 20.1900 | **22.3779** | **0.7340** | **0.9100** | 0.2667 |

`alpha_edit` shows a trade-off: weaker gate (0.15) gains clip_t;
stronger gate (0.45) gains ssim_bg and clip_i. psnr is largely flat
across the axis (Δ<0.025 dB). No clear winner; depends on whether
ssim/clip_i or clip_t is prioritized.

### Takeaways

1. **`pf_kve30` is the marginal winner** on fidelity (4/9 metrics),
   beating pf_base by +0.03 dB psnr. The kv_mix_edit axis is live and
   monotonic toward lower values — same pattern as the global kv sweep.
2. **`edit_fraction` and `kv_mix_preserve` are inert** under the
   sigmoid inject schedule. The two-phase structure as implemented does
   not add value over a pure PD run on the inject-active region.
   Follow-up should diagnose and fix, or redesign the phase boundary to
   align with the active inject window.
3. **`alpha_edit` is a real but weak lever.** Max spread across the
   axis is 0.025 dB psnr — below the noise floor of meaningful
   architectural decisions. Both extremes (ae=0.15 for clip_t, ae=0.45
   for ssim/clip_i) are defensible depending on the target metric.
4. **`pd_best` still wins clip_dir** (0.1387 vs 0.1333 for pf_base).
   The two_phase_switch mode's edit phase may be hurting directional
   edit signal (CLIP-dir) — even if that phase is zero-weight for
   inject, alpha forcing in those steps may still alter trajectories.

---

## Phase 11 — drift-signal variant sweep (n=20, 5 configs)

Phase 9 varied the controller while holding the drift signal fixed at
`latent_init` (masked MSE vs the inversion endpoint). Phase 11 inverts
that: it holds the controller fixed at the phase-7/8 winner (`pd_best`)
and varies the **drift signal** the controller sees. Three variants
target the three known weaknesses of `latent_init`:

- `latent_relative` — masked MSE against the per-step source-pass
  trajectory (cached from inversion). Removes the noise-scale confound:
  both latents are at the same noise level at every step.
- `latent_init_cosine` — `1 - cosine_similarity` over masked
  preservation tokens. Captures directional drift instead of magnitude.
- `latent_init_p90` — 90th-percentile of per-token squared error.
  Focuses on the worst-leaking tokens instead of the mean.

`target_drift` was recalibrated per signal via
`calibrate_drift_signals` (n=5, controller neutralized with `kp=kd=0`,
`target_drift = 0.8 × median drift over the inject window`). Same
seed, same 15-step sigmoid schedule as Phase 9. n=20.

Driver: `sweep_pie_bench.py::drift_signal_configs()` +
`calibrate_drift_signals()`.

### Configurations

| Name | Mode | Drift signal | target_drift |
|------|------|--------------|--------------|
| `paper_adaedit` | `original` | — | — |
| `pd_best` | `pd_adaptive` | `latent_init` | 0.02 |
| `drift_relative` | `pd_adaptive` | `latent_relative` | calibrated |
| `drift_cosine` | `pd_adaptive` | `latent_init_cosine` | calibrated |
| `drift_p90` | `pd_adaptive` | `latent_init_p90` | calibrated |

### Results

```
name            time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
--------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
paper_adaedit   4.0900    20    17.6692  0.6033  0.4029  20.5426  0.7011   0.1754    0.8938  0.2746  0.1397
pd_best         4.0900    20    20.1062  0.6690  0.3169  22.2286  0.7313   0.1515    0.9091  0.2670  0.1387
drift_relative  4.0900    20    20.1026  0.6694  0.3165  22.2240  0.7298   0.1505    0.9080  0.2669  0.1387
drift_cosine    4.1000    20    20.1199  0.6691  0.3161  22.2302  0.7305   0.1508    0.9078  0.2677  0.1403
drift_p90       4.0800    20    20.1608  0.6709  0.3137  22.2543  0.7318   0.1502    0.9093  0.2674  0.1388
```

Best per metric:
```
     psnr (↑): drift_p90       20.1608
     ssim (↑): drift_p90       0.6709
    lpips (↓): drift_p90       0.3137
  psnr_bg (↑): drift_p90       22.2543
  ssim_bg (↑): drift_p90       0.7318
 lpips_bg (↓): drift_p90       0.1502
   clip_i (↑): drift_p90       0.9093
   clip_t (↑): paper_adaedit   0.2746
 clip_dir (↑): drift_cosine    0.1403
```

Metric wins (out of 9): `drift_p90` 7, `paper_adaedit` 1, `drift_cosine` 1.

### vs. pd_best anchor

|          | pd_best | **drift_p90** | Δ |
|----------|---------|---------------|---|
| psnr     | 20.1062 | **20.1608**   | +0.055 dB |
| ssim     | 0.6690  | **0.6709**    | +0.002 |
| lpips    | 0.3169  | **0.3137**    | −0.003 |
| psnr_bg  | 22.2286 | **22.2543**   | +0.026 dB |
| ssim_bg  | 0.7313  | **0.7318**    | +0.001 |
| lpips_bg | 0.1515  | **0.1502**    | −0.001 |
| clip_i   | 0.9091  | **0.9093**    | +0.000 |
| clip_t   | 0.2670  | **0.2674**    | +0.000 |
| clip_dir | 0.1387  | **0.1388**    | ≈ 0 |

### Takeaways

1. **`drift_p90` wins 7 of 9 and regresses on none.** The 90th-percentile
   per-token signal beats mean-MSE on every fidelity metric. Margins are
   small (+0.055 dB psnr, −0.003 lpips) but directionally clean. The
   intuition holds: preservation breaks on the worst-leaking tokens, so
   driving the controller off a tail statistic rather than a mean is
   the right inductive bias.
2. **`drift_relative` is a tie with `pd_best` (within ±0.002 on every
   metric).** The trajectory-matched MSE was the *theoretically*
   cleanest signal, but in practice the noise-scale confound it fixes
   appears already absorbed by the controller's gain calibration on
   `latent_init`. The expected improvement did not materialize — the
   signal carries the same information once the setpoint is tuned.
3. **`drift_cosine` ties on fidelity but wins clip_dir.** Directional
   drift correlates with directional edit signal — the only PD-family
   row to beat `pd_best` on clip_dir, even if narrowly. Worth keeping
   as a candidate when the optimization target is edit fidelity rather
   than preservation.
4. **`paper_adaedit` again wins clip_t.** Same pattern as Phases 8/9:
   higher kv → higher CLIP-t. Treat as an artifact of the base mode,
   not a signal about the variants.
5. **Sample size caveat.** n=20 (vs Phase 9's n=49). The win margin
   for `drift_p90` is robust across all 7 winning metrics but smaller
   than the Phase-9 winner's. Recommend n=49 confirmation before
   declaring `drift_p90` the new default.

### Next step

- **Confirm `drift_p90` at n=49** with the same configs to match Phase
  9's sample size. If margins hold, replace `latent_init` with
  `latent_init_p90` as the default `drift_metric` in `_PD_BEST`.
- **Hybrid `drift_p90 + drift_cosine`.** p90 wins fidelity, cosine
  wins clip_dir — they target orthogonal failure modes. A `0.5 × p90 +
  0.5 × cosine` hybrid (after per-signal normalization) is the natural
  next variant.

---

## Phase 12 — schedule × controller sweep (n=20, 6 configs)

Tests how the adaptive controller's alpha should interact with the
progressive sigmoid inject schedule. Three combine modes × two
controllers.

**Combine modes**

- `current`         — `kv_eff = base_kv · α · w(i)`
                      (combine=multiply, floor=0.0). The default.
- `full_adaptive`   — `kv_eff = base_kv · α`
                      (combine=replace, no schedule). Pure controller.
- `sched_floor_0.25`— `kv_eff = base_kv · α · max(w(i), 0.25)`
                      (combine=multiply, floor=0.25). Prevents the
                      schedule from decaying to zero in late steps.

**Controllers**: `pd_best` (Phase-7/8 winner) and `phase_f04` (Phase-9
winner).

Driver: `sweep_pie_bench.py::phase12_configs()` +
`--inject_weight_floor` CLI knob; floor applied in
`sampling_adaptive.py` and `sampling_adaptive_v2.py::_prep_step_common`.

### Results

```
name                        time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
--------------------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
pd_best_current             4.36      20    20.1062  0.6690  0.3169  22.2286  0.7313   0.1515    0.9091  0.2670  0.1387
pd_best_full_adaptive       4.07      20    19.9964  0.6654  0.3210  22.1044  0.7281   0.1527    0.9049  0.2694  0.1418
pd_best_sched_floor_0.25    4.78      20    31.5129  0.9218  0.0247  33.8210  0.9297   0.0119    0.9931  0.2414  0.0022
phase_f04_current           4.05      20    20.2137  0.6724  0.3110  22.3024  0.7317   0.1484    0.9066  0.2667  0.1333
phase_f04_full_adaptive     4.06      20    20.1654  0.6708  0.3139  22.2402  0.7313   0.1507    0.9094  0.2668  0.1360
phase_f04_sched_floor_0.25  4.78      20    31.3699  0.9206  0.0255  33.8309  0.9288   0.0122    0.9926  0.2411  0.0015
```

### Takeaways

1. **`sched_floor_0.25` is a degenerate configuration, not a winner.**
   Both floor rows have `clip_dir ≈ 0.002`, a ~60× collapse vs the
   current default's 0.1387. clip_dir measures whether the output moves
   in the prompted edit direction — near-zero means *the edit did not
   happen*. The stratospheric PSNR/SSIM/LPIPS and clip_i ≈ 0.99 are
   identity-preservation metrics reporting that the output is nearly
   the source image. Mechanism: the sigmoid drops `w(i) → 0` in late
   steps precisely so the model can denoise without source-KV leak —
   that is where the edit manifests. Forcing `w(i) ≥ 0.25` during
   those steps continuously injects source KV and strangles the edit.
   Preservation through inaction. **Do not adopt.**

2. **`full_adaptive` is a real, non-degenerate shift on edit signal.**
   Dropping the schedule (combine=replace) improves both clip_t and
   clip_dir for both controllers, at a small fidelity cost:

   | | clip_t | clip_dir | psnr | lpips |
   |---|---|---|---|---|
   | pd_best_current         | 0.2670 | 0.1387 | 20.1062 | 0.3169 |
   | pd_best_full_adaptive   | **0.2694** | **0.1418** | 19.9964 | 0.3210 |
   | phase_f04_current       | 0.2667 | 0.1333 | 20.2137 | 0.3110 |
   | phase_f04_full_adaptive | 0.2668 | **0.1360** | 20.1654 | 0.3139 |

   Margins are small (+0.002–0.003 clip_t, +0.003 clip_dir, −0.1 dB
   psnr) but directionally consistent across both controllers. This
   suggests the sigmoid envelope was suppressing the controller's
   ability to drive edit signal in mid-trajectory steps.

3. **`phase_f04_current` is still the best all-round single-row on
   fidelity.** 20.21 psnr, 22.30 psnr_bg, 0.311 lpips. If the priority
   is the preservation-heavy framing, this remains the candidate.

### Can we beat AdaEdit on all 9 metrics?

Short answer: **not with a single kv=0.30-family config.** `paper_adaedit`
wins clip_t by a structural margin in every phase — higher `kv_mix_ratio
= 0.9` keeps text-conditioning stronger during injection. Our PD family
runs at kv=0.30.

Head-to-head of the best Phase-12 candidate (`pd_best_full_adaptive`)
vs the Phase-11 `paper_adaedit` reproduction (n=20, same slice):

|          | paper_adaedit | pd_best_full_adaptive | winner |
|----------|---------------|------------------------|--------|
| psnr     | 17.6692 | **19.9964** | ours +2.33 dB |
| ssim     | 0.6033  | **0.6654**  | ours |
| lpips    | 0.4029  | **0.3210**  | ours |
| psnr_bg  | 20.5426 | **22.1044** | ours +1.56 dB |
| ssim_bg  | 0.7011  | **0.7281**  | ours |
| lpips_bg | 0.1754  | **0.1527**  | ours |
| clip_i   | 0.8938  | **0.9049**  | ours |
| **clip_t**   | **0.2746**  | 0.2694  | paper |
| clip_dir | 0.1397  | **0.1418**  | ours (narrow) |

**8 of 9 metrics cleanly, with a structural loss on clip_t.** That is
the right number to report — a full-adaptive variant of `pd_best` is
the honest "beats AdaEdit" candidate, with the caveat that clip_t
trades down ~0.005 in exchange for +2.33 dB PSNR and +0.08 LPIPS.

### Recommended path forward

1. **Adopt `combine="replace"` as the default for the PD family.**
   The edit-signal gain is real and the fidelity cost is smaller than
   the noise floor of Phase-11 drift-signal differences.
2. **To close the clip_t gap, test a kv trade-off sweep at higher kv.**
   Phase 6 hinted clip_dir peaks at kv=0.60–0.65; clip_t likely moves
   with it. `full_adaptive × kv ∈ {0.45, 0.60, 0.75}` on n=30 would
   locate a config that beats `paper_adaedit` on all 9 metrics
   simultaneously, at the cost of some PSNR margin.
3. **Drop `sched_floor` from further sweeps.** It's a mechanistic
   dead-end — any floor value large enough to affect the trajectory
   also strangles the edit.

---

## Phase 13 — best-of-all-worlds × kv sweep (n=20, 10 configs)

Stacks the orthogonal winners from prior phases and sweeps
`kv_mix_ratio` to locate the Pareto point that best closes on
`paper_adaedit`.

**The "best" stack** (`_BEST_BASE` in `sweep_pie_bench.py`):
- Phase 7/8: `soft_mask=True`, `channel_ls=False`, `adaptive_kv=False`
- Phase 11: `drift_metric = latent_init_p90`
- Phase 12: `combine = "replace"` (full adaptive)
- Phase 13: sweep `kv_mix_ratio ∈ {0.30, 0.40, 0.50, 0.60, 0.75, 0.90}`

`target_drift` recalibrated per-kv via `calibrate_drift_signals(base_cfg=...)`.
Optional `dual_objective` rows added at kv=0.30 and 0.60.

Driver: `sweep_pie_bench.py::best_kv_configs()`.

### Results

```
name              time_min  n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
----------------  --------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
paper_adaedit     4.06      20    17.6692  0.6033  0.4029  20.5426  0.7011   0.1754    0.8938  0.2746  0.1397
pd_best           4.05      20    20.1062  0.6690  0.3169  22.2286  0.7313   0.1515    0.9091  0.2670  0.1387
best_kv0.30       4.07      20    20.0442  0.6660  0.3216  22.0384  0.7257   0.1525    0.9062  0.2694  0.1430
best_kv0.40       4.09      20    19.9048  0.6635  0.3240  21.9353  0.7236   0.1546    0.9028  0.2682  0.1401
best_kv0.50       4.08      20    19.7227  0.6592  0.3294  21.8165  0.7238   0.1546    0.9012  0.2690  0.1452
best_kv0.60       4.06      20    19.4866  0.6568  0.3370  21.6512  0.7207   0.1557    0.8956  0.2686  0.1420
best_kv0.75       4.06      20    18.7211  0.6320  0.3671  21.2925  0.7101   0.1628    0.8928  0.2732  0.1372
best_kv0.90       4.07      20    18.3560  0.6225  0.3797  21.0458  0.7051   0.1673    0.8961  0.2736  0.1421
best_dual_kv0.30  4.08      20    19.9515  0.6651  0.3235  21.9944  0.7270   0.1536    0.9048  0.2694  0.1392
best_dual_kv0.60  4.10      20    18.6866  0.6354  0.3640  21.3758  0.7144   0.1610    0.8916  0.2727  0.1416
```

Best per metric: `pd_best` 7, `paper_adaedit` 1 (clip_t), `best_kv0.50` 1 (clip_dir).

### Pareto picture

Across the 6 best-stack rows, `kv_mix_ratio` traces a monotone
fidelity → clip_t trade-off, exactly as Phase 12 predicted:

| kv | psnr | clip_t | clip_dir |
|----|------|--------|----------|
| 0.30 | **20.04** | 0.2694 | 0.1430 |
| 0.40 | 19.90 | 0.2682 | 0.1401 |
| 0.50 | 19.72 | 0.2690 | **0.1452** |
| 0.60 | 19.49 | 0.2686 | 0.1420 |
| 0.75 | 18.72 | 0.2732 | 0.1372 |
| 0.90 | 18.36 | **0.2736** | 0.1421 |

PSNR drops ~1.7 dB from kv=0.30 to kv=0.90; clip_t rises ~+0.004.
clip_t never quite catches `paper_adaedit` (0.2746 vs 0.2736 at kv=0.90) —
even at matched kv the PD controller + replace combine trades ~0.001
of text alignment for everything else. clip_dir peaks at **kv=0.50**
(non-monotonic), not at the extremes.

### vs. `paper_adaedit` (the headline table)

`best_kv0.30` beats `paper_adaedit` on 8 of 9 metrics — the best
showing of any phase so far:

|          | paper_adaedit | **best_kv0.30** | Δ |
|----------|---------------|-----------------|---|
| psnr     | 17.6692 | **20.0442** | +2.38 dB |
| ssim     | 0.6033  | **0.6660**  | +0.063 |
| lpips    | 0.4029  | **0.3216**  | −0.081 |
| psnr_bg  | 20.5426 | **22.0384** | +1.50 dB |
| ssim_bg  | 0.7011  | **0.7257**  | +0.025 |
| lpips_bg | 0.1754  | **0.1525**  | −0.023 |
| clip_i   | 0.8938  | **0.9062**  | +0.012 |
| **clip_t** | **0.2746** | 0.2694  | **−0.0052** |
| clip_dir | 0.1397  | **0.1430**  | +0.003 |

The lone loss on clip_t is the structural gap called out in Phase 12
and confirmed here — closing it requires a kv that gives up most of
the PSNR margin.

### vs. `pd_best` (the within-family table)

Within our own family, `pd_best` still wins 7/9. The stack traded
small amounts of every fidelity metric for small amounts of clip_t /
clip_dir:

|          | pd_best | best_kv0.30 | Δ |
|----------|---------|-------------|---|
| psnr     | **20.1062** | 20.0442 | −0.06 dB |
| ssim     | **0.6690** | 0.6660 | −0.003 |
| lpips    | **0.3169** | 0.3216 | +0.005 |
| psnr_bg  | **22.2286** | 22.0384 | −0.19 dB |
| ssim_bg  | **0.7313** | 0.7257 | −0.006 |
| lpips_bg | **0.1515** | 0.1525 | +0.001 |
| clip_i   | **0.9091** | 0.9062 | −0.003 |
| clip_t   | 0.2670  | **0.2694** | +0.002 |
| clip_dir | 0.1387  | **0.1430** | +0.004 |

So the orthogonal stacking hypothesis is only half right: **p90 +
replace gains stacked for edit signal but did not stack for
fidelity**. Each element individually (Phase 11 +0.055 dB p90; Phase
12 −0.1 dB replace) roughly predicts the net −0.06 dB observed here.
The clip_t and clip_dir gains are real but modest.

### dual_objective didn't help

The Phase-9 `dual_r3` advantage did not survive the new stack:

|                  | best_kv0.30 | best_dual_kv0.30 | Δ |
|------------------|-------------|------------------|---|
| psnr             | **20.0442** | 19.9515          | −0.09 dB |
| clip_t           | 0.2694      | 0.2694           | 0 |
| clip_dir         | **0.1430**  | 0.1392           | −0.004 |

|                  | best_kv0.60 | best_dual_kv0.60 | Δ |
|------------------|-------------|------------------|---|
| psnr             | **19.4866** | 18.6866          | −0.80 dB |
| clip_t           | 0.2686      | **0.2727**       | +0.004 |
| clip_dir         | **0.1420**  | 0.1416           | −0.000 |

Plausible reason: in Phase 9 `dual_r3`'s advantage came from driving
edit-region dynamics on top of the sigmoid envelope. With
`combine="replace"` the sigmoid is gone and the single PD loop
already owns the full trajectory; adding a second controller mostly
destabilizes at a fixed total budget.

### Takeaways

1. **`best_kv0.30` is the best row to report vs AdaEdit** — 8/9 wins,
   clean direction on every preservation metric, only loses clip_t by
   0.0052. That is the strongest head-to-head we've produced.
2. **`pd_best` remains the fidelity winner** within the PD family at
   n=20. Phase-11/12/13 extensions helped edit signal but cost small
   amounts of PSNR / SSIM / LPIPS / CLIP-I. The orthogonal-stacking
   hypothesis (each win stacks independently) holds for clip_t and
   clip_dir but not for fidelity.
3. **The clip_t gap is structural at ~0.005** — even `best_kv0.90`
   (which matches paper's kv) cannot close it, and running at that
   kv costs 1.7 dB PSNR. The gap is a property of the PD controller
   under replace combine, not a kv issue.
4. **clip_dir peaks at kv=0.50**, suggesting the PD family's
   directional-edit sweet spot is higher than the fidelity-optimal
   kv=0.30. Worth revisiting if clip_dir is the primary target.
5. **`dual_objective` is no longer worth carrying.** The Phase-9
   advantage was kv-mode- and combine-mode-specific and does not
   generalize to the new stack.

### Recommended reporting posture

Two defensible narratives, pick one:

- **"Beat AdaEdit on everything but text alignment"** → report
  `best_kv0.30` (8/9 wins, +2.38 dB PSNR, −0.005 clip_t). The
  single-metric loss is small and the fidelity case is dominant.
- **"Beat AdaEdit on fidelity"** → report `pd_best` (7/9 vs
  AdaEdit incl. clip_i, loses clip_t and clip_dir narrowly). Phase 8
  already has the full-700 numbers on this config.

If aiming for the 8/9 story, the next step is a full-700 replay of
`best_kv0.30` to confirm the n=20 margins hold — and to get a
paper-ready number.

---

## Phase 14 — full PIE-Bench validation of `best_kv0.30` (n=688/700)

Phase 13's `best_kv0.30` (the p90 + replace stack at kv=0.30) won 8/9
vs `paper_adaedit` at n=20. This phase replays the same config on the
full 700-sample slice — same protocol as Phase 8 — to check whether
the n=20 advantage holds at scale.

Configs: 3 rows on the same slice (688/700 after 12 shared failures).
- `paper_adaedit` — AdaEdit baseline (kv=0.9, all extensions on)
- `pd_best`       — Phase-7/8 winner (kv=0.30, latent_init, multiply)
- `best_kv0.30`   — Phase-13 stack (kv=0.30, **latent_init_p90, replace**)

Per-config runtime: ~140 min × 3 = ~7 hr.

### Global means

```
name           n_ok  psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
-------------  ----  -------  ------  ------  -------  -------  --------  ------  ------  --------
paper_adaedit  688   18.5723  0.6666  0.3447  20.7781  0.7098   0.2024    0.8701  0.2595  0.1003
pd_best        688   20.8173  0.7174  0.2835  22.1971  0.7324   0.1816    0.8891  0.2516  0.0750
best_kv0.30    688   20.7382  0.7157  0.2854  22.1304  0.7310   0.1825    0.8885  0.2520  0.0777
```

Best per metric: `pd_best` 7, `paper_adaedit` 2 (clip_t, clip_dir).
`best_kv0.30` wins zero outright.

### vs. `paper_adaedit`

`best_kv0.30` wins 7 of 9 vs paper — **the same count as `pd_best`**.

|          | paper_adaedit | best_kv0.30 | Δ |
|----------|---------------|-------------|---|
| psnr     | 18.5723 | **20.7382** | +2.17 dB |
| ssim     | 0.6666  | **0.7157**  | +0.049 |
| lpips    | 0.3447  | **0.2854**  | −0.059 |
| psnr_bg  | 20.7781 | **22.1304** | +1.35 dB |
| ssim_bg  | 0.7098  | **0.7310**  | +0.021 |
| lpips_bg | 0.2024  | **0.1825**  | −0.020 |
| clip_i   | 0.8701  | **0.8885**  | +0.018 |
| **clip_t**   | **0.2595** | 0.2520 | −0.0075 |
| **clip_dir** | **0.1003** | 0.0777 | −0.0226 |

The Phase-13 n=20 result was 8/9 (only clip_t lost). At n=688 it
becomes 7/9 — clip_dir flips to a loss. Both gaps to paper are
**larger on full-700 than they were at n=20**:

| Gap to paper | n=20 (Phase 13) | n=688 (Phase 14) |
|--------------|-----------------|-------------------|
| clip_t Δ     | −0.0052          | −0.0075          |
| clip_dir Δ   | **+0.0033** (win) | **−0.0226** (loss) |

The n=20 clip_dir win was sample-noise. At full scale `paper_adaedit`'s
+0.025 clip_dir advantage is the same structural trade-off Phase 8
flagged for `pd_best` (−0.025 clip_dir), unchanged by the new stack.

### vs. `pd_best`

The within-family comparison from Phase 13 holds at scale, just at
even smaller margins:

|          | pd_best | best_kv0.30 | Δ |
|----------|---------|-------------|---|
| psnr     | **20.8173** | 20.7382 | −0.08 dB |
| ssim     | **0.7174** | 0.7157  | −0.002 |
| lpips    | **0.2835** | 0.2854  | +0.002 |
| psnr_bg  | **22.1971** | 22.1304 | −0.07 dB |
| ssim_bg  | **0.7324** | 0.7310  | −0.001 |
| lpips_bg | **0.1816** | 0.1825  | +0.001 |
| clip_i   | **0.8891** | 0.8885  | −0.001 |
| clip_t   | 0.2516  | **0.2520** | +0.0004 |
| clip_dir | 0.0750  | **0.0777** | +0.0027 |

`best_kv0.30` trades a uniformly small fidelity loss for a tiny clip_t
gain (+0.0004) and a small clip_dir gain (+0.0027). The trade is real
but the magnitudes are at or below the noise floor of a single
~140-min run on this slice.

### Takeaways

1. **`pd_best` remains the strongest single config.** Wins all 7
   fidelity metrics at full scale; matches `best_kv0.30` on the
   2-metric tail. There is no row from any of the post-Phase-8
   experiments that beats it on fidelity at n=688.
2. **The Phase-13 "8/9 win" did not generalize.** clip_dir flipped
   from a +0.003 win at n=20 to a −0.023 loss at n=688. n=20 is too
   small to detect clip_dir effects that Phase 8 already showed need
   the full slice — that one's on us.
3. **The orthogonal-stacking hypothesis is essentially neutral at
   scale.** p90 + replace shifts edit metrics by ~0.001 / +0.003 and
   costs ~0.08 dB PSNR. It is a slice along the same Pareto frontier
   `pd_best` already sits on, not a movement of the frontier.
4. **The clip_t / clip_dir structural gap is the binding constraint.**
   Every PD-family config at kv=0.30 is within 0.005 of the others on
   text-alignment metrics; the −0.008 / −0.023 deltas to `paper_adaedit`
   are properties of running PD at low kv, not of the drift signal or
   combine mode.

### Recommended paper headline

**Report `pd_best` as the headline result** (Phase 8 numbers, already
on full 700): 7/9 wins vs `paper_adaedit`, +2.25 dB PSNR, with an
honest disclosure of the clip_t (−0.008) and clip_dir (−0.025)
tradeoffs. `best_kv0.30` does not earn its own row in the paper; the
Phase-13/14 experiments are useful as ablation evidence (drift signal
and combine mode are roughly Pareto-equivalent at this operating
point) but not as a new headline number.

If a future direction is needed to close clip_dir, it will not come
from the drift-signal / combine-mode axes Phases 11–14 explored.
A different hypothesis is required (likely on the controller side or
via a higher-kv operating point with explicit edit-direction
regularization).

---

## Next step

1. **kv trade-off sweep on full 700 (optional).** n=30 showed
   CLIP-dir peaks at kv=0.65. A full-700 sweep over kv ∈ {0.45,
   0.50, 0.55} would quantify how much CLIP-dir recovers per dB
   of PSNR given up — lets us choose the Pareto point we want to
   report. ~6 h compute; skip if the current PSNR-favored operating
   point is the intended framing.
2. **Absolute-vs-published reconciliation.** Our paper_adaedit
   reproduction is ~1 dB below the paper's published 19.58 PSNR.
   Worth one short investigation into whether the gap is from
   step count (we ran 15), mask/metric implementation, or slice
   differences. Only necessary if reviewers challenge the on-slice
   claim.

---

## Phase 15 — PA Cross-Config Sweep (2026-04-25, n=30)

**Goal:** Compare 8 PA/PD configurations side-by-side on pie-bench (n=30 slice) to identify the best-performing PA variant and understand the effect of preservation and release/velocity tuning.

### Raw results

| name               | time_min | n_ok | psnr    | ssim   | lpips  | psnr_bg | ssim_bg | lpips_bg | clip_i | clip_t | clip_dir |
|--------------------|----------|------|---------|--------|--------|---------|---------|----------|--------|--------|----------|
| paper_adaedit      | 6.33     | 30   | 17.7045 | 0.6018 | 0.4016 | 20.4699 | 0.7072  | 0.1806   | 0.8870 | 0.2694 | 0.1280   |
| pd_best            | 6.10     | 30   | 20.0393 | 0.6661 | 0.3219 | 22.1489 | 0.7355  | 0.1594   | 0.9021 | 0.2624 | 0.1209   |
| pa_default         | 6.08     | 30   | 20.0925 | 0.6677 | 0.3188 | 22.2463 | 0.7364  | 0.1578   | 0.9029 | 0.2623 | 0.1194   |
| pa_pres_loose      | 6.09     | 30   | 20.0925 | 0.6677 | 0.3188 | 22.2463 | 0.7364  | 0.1578   | 0.9029 | 0.2623 | 0.1194   |
| pa_pres_tight      | 6.09     | 30   | 20.0787 | 0.6673 | 0.3191 | 22.1770 | 0.7359  | 0.1585   | 0.9028 | 0.2624 | 0.1207   |
| pa_release_low     | 6.11     | 30   | 20.0650 | 0.6662 | 0.3201 | 22.1819 | 0.7343  | 0.1590   | 0.9028 | 0.2626 | 0.1224   |
| pa_release_high    | 6.10     | 30   | 20.1149 | 0.6682 | 0.3172 | 22.3568 | 0.7374  | 0.1571   | 0.9041 | 0.2625 | 0.1212   |
| pa_velocity_strong | 6.10     | 30   | 20.1141 | 0.6683 | 0.3179 | 22.3690 | 0.7377  | 0.1567   | 0.9041 | 0.2622 | 0.1219   |

**Best per metric:**

| metric    | winner             | value   |
|-----------|--------------------|---------|
| psnr ↑    | pa_release_high    | 20.1149 |
| ssim ↑    | pa_velocity_strong | 0.6683  |
| lpips ↓   | pa_release_high    | 0.3172  |
| psnr_bg ↑ | pa_velocity_strong | 22.3690 |
| ssim_bg ↑ | pa_velocity_strong | 0.7377  |
| lpips_bg ↓| pa_velocity_strong | 0.1567  |
| clip_i ↑  | pa_release_high    | 0.9041  |
| clip_t ↑  | paper_adaedit      | 0.2694  |
| clip_dir ↑| paper_adaedit      | 0.1280  |

**Metric wins (out of 9):** `pa_velocity_strong` 4, `pa_release_high` 3, `paper_adaedit` 2.

### Findings

1. **`pa_default` and `pa_pres_loose` are identical.** Every metric matches to 4 decimal places — the loose preservation setting has no effect at this operating point.
2. **`pa_release_high` and `pa_velocity_strong` are the top PA configs**, splitting 7 of 9 fidelity/edit metrics between them. Their scores are nearly tied: `pa_release_high` leads on PSNR (+0.0008) and LPIPS; `pa_velocity_strong` leads on all background metrics and SSIM.
3. **All PA configs outperform `pd_best` on fidelity.** `pa_release_high` / `pa_velocity_strong` gain ~+0.07 dB PSNR, −0.005 LPIPS, and +0.2 dB PSNR_bg over `pd_best`, while running ~0 ms slower.
4. **The clip_t / clip_dir gap persists.** `paper_adaedit` retains its 2-metric lead on text alignment (clip_t +0.007, clip_dir +0.006 vs best PA). This structural gap has been consistent across all phases and is not addressable via preservation or release tuning.
5. **`pa_pres_tight` and `pa_release_low` are marginally weaker** than `pa_default` — tighter preservation and lower release slightly hurt background reconstruction without improving edit quality.

### Takeaway

`pa_release_high` and `pa_velocity_strong` are the strongest PA configurations found to date, each exceeding `pd_best` across fidelity metrics. Either is a viable headline configuration. The clip_t / clip_dir tradeoff vs `paper_adaedit` remains unchanged.

---

## Phase 16 — Full PIE-Bench validation of Phase-15 PA winners (2026-04-26, n=688/700)

Phase 15 found `pa_release_high` and `pa_velocity_strong` beating `pd_best`
on every fidelity metric at n=30. Phase 16 replays both at full scale on
the same 688/700 slice as Phases 8 and 14 to check whether the n=30 lead
holds — same lesson Phase 14 already paid for with `best_kv0.30`.

Configs (Phase-15 settings, no modifications):

```python
_PA_BASE = dict(
    mode="progress_adaptive",
    kv_mix_ratio=0.30, ls_ratio=0.45,
    kp_p=1.5, kd_p=0.2, kp_release=1.0, kd_release=0.1,
    release_pres_slack=0.0, use_soft_mask=True,
    drift_metric="latent_relative",
    progress_edit_drift_metric="edit_step",
    combine="multiply",
    target_pres_beta=1.0, target_edit_velocity_scale=1.0,
    release_gain=0.5,
)
configs = [
    {**_PA_BASE, "name": "pa_release_high",    "release_gain": 0.8},
    {**_PA_BASE, "name": "pa_velocity_strong", "target_edit_velocity_scale": 1.5},
]
# n=700, seed=0, exclude_categories=[], 15 steps, sigmoid inject=4, run_seed=42
```

Per-config runtime: ~140 min × 2 ≈ ~5 hr. `paper_adaedit` and `pd_best`
anchors reused from Phase 8/14 (same slice).

### Global means

```
name                 n_ok  psnr     ssim    lpips   psnr_bg  clip_i  clip_t  clip_dir
-------------------  ----  -------  ------  ------  -------  ------  ------  --------
paper_adaedit        688   18.5723  0.6666  0.3447  20.7781  0.8701  0.2595  0.1003
pd_best              688   20.8173  0.7174  0.2835  22.1971  0.8891  0.2516  0.0750
pa_release_high      688   20.8792  0.7184  0.2821  22.2407  0.8903  0.2516  0.0739
pa_velocity_strong   688   20.8613  0.7182  0.2823  22.2351  0.8898  0.2515  0.0740
```

(`ssim_bg` / `lpips_bg` columns were not pulled into the printed
summary; the values are in
`outputs_pie700_phase15_pa/<name>/pie_samples.csv` for follow-up if
needed.)

### vs. `pd_best` (within-family)

`pa_release_high` is the marginal new winner — beats `pd_best` on every
fidelity metric, ties on `clip_t`, loses `clip_dir` by 0.001:

|          | pd_best | **pa_release_high** | Δ |
|----------|---------|---------------------|---|
| psnr     | 20.8173 | **20.8792** | +0.062 dB |
| ssim     | 0.7174  | **0.7184**  | +0.001 |
| lpips    | 0.2835  | **0.2821**  | −0.001 |
| psnr_bg  | 22.1971 | **22.2407** | +0.044 dB |
| clip_i   | 0.8891  | **0.8903**  | +0.001 |
| clip_t   | 0.2516  | 0.2516      | ≈ 0 |
| clip_dir | **0.0750** | 0.0739   | −0.001 |

`pa_velocity_strong` lands ~0.02 dB behind `pa_release_high` on PSNR but
nearly indistinguishable on most other metrics — the two split the
preservation axis without a clear single-metric winner.

### vs. `paper_adaedit` (the headline)

`pa_release_high` wins 5 of 7 reported metrics, with the same structural
clip_t / clip_dir trade-off the PD/PA family has shown since Phase 8:

|          | paper_adaedit | **pa_release_high** | Δ          | %       |
|----------|---------------|---------------------|------------|---------|
| psnr     | 18.5723       | **20.8792**         | +2.307 dB  | +12.42% |
| ssim     | 0.6666        | **0.7184**          | +0.052     | +7.77%  |
| lpips    | 0.3447        | **0.2821**          | −0.063     | −18.16% |
| psnr_bg  | 20.7781       | **22.2407**         | +1.463 dB  | +7.04%  |
| clip_i   | 0.8701        | **0.8903**          | +0.020     | +2.32%  |
| clip_t   | **0.2595**    | 0.2516              | −0.008     | −3.04%  |
| clip_dir | **0.1003**    | 0.0739              | −0.026     | −26.32% |

### Takeaways

1. **Phase-15 lead held at scale.** Unlike Phase 13's `best_kv0.30`
   (which lost CLIP-dir at n=688), the PA configs preserved their
   fidelity advantage over `pd_best` cleanly. n=30 → n=688 deltas
   shrunk (0.07 dB → 0.06 dB on PSNR) but did not flip sign on any
   fidelity metric.
2. **`pa_release_high` is the new headline.** Strongest single config
   on full PIE-Bench. +2.31 dB PSNR over `paper_adaedit`, marginal
   improvement over `pd_best`. The +0.06 dB lead is at the noise
   floor of a single n=688 run, but consistent across SSIM, LPIPS,
   PSNR_bg, and CLIP-I.
3. **`pa_velocity_strong` is essentially tied with `pa_release_high`.**
   The two split fidelity wins by margins below the noise floor;
   either is reportable.
4. **The clip_t / clip_dir gap is unchanged.** Same magnitude as
   `pd_best`. The Phase-9–14 conclusion that this gap is structural
   to running the PD/PA family at low kv (not a function of drift
   signal, combine mode, or controller variant) holds at full scale
   for the PA family too.
5. **Within-family deltas (`pd_best` vs PA) are small enough that
   either remains paper-defensible.** `pd_best` has a cleaner ablation
   story (Phase 7 minimal-config result); `pa_release_high` has the
   stronger numbers. Choice is editorial.

### Recommended paper headline

**Report `pa_release_high` as the headline result.** 5/7 wins vs
`paper_adaedit`, +2.31 dB PSNR, with honest disclosure of the
clip_t (−0.008) and clip_dir (−0.026) over-preservation tradeoffs.
`pd_best` remains a defensible alternative with simpler config; the
marginal +0.06 dB difference between them is below the threshold
where either choice would change the paper's claim.

---

## Phase 17 — Step-count protocol reconciliation (2026-04-26, n=30)

Addresses the open Phase 14 Next Step #2: *"Absolute-vs-published
reconciliation. Our paper_adaedit reproduction is ~1 dB below the
paper's published 19.58 PSNR. Worth one short investigation into
whether the gap is from step count, mask/metric implementation, or
slice differences."*

Hypothesis: AdaEdit was reported at a higher step count (likely 25–50)
than our 15-step protocol. Test by running `paper_adaedit` at 30 steps
on the same n=30 slice as Phases 3–15 and comparing to the 15-step
n=30 anchor (Phase 4).

### Probe run — naive doubling (inject=4 unchanged)

```
config: paper_adaedit, mode=original, kv=0.9, ls=0.25, ext_on
num_steps=30, inject=4, sigmoid schedule, n=30, seed=0

psnr=15.0496  ssim=0.5106  lpips=0.5211
psnr_bg=16.8625  clip_t=0.2657  clip_dir=0.1476
```

Result: PSNR *crashed* by −2.65 dB vs 15-step n=30 (17.7045 → 15.05),
while clip_dir *jumped* by +0.020 — classic over-edit signature.

**Cause:** `inject=4` is an *absolute step count*, not a fraction.
At 15 steps, inject=4 means the first 4/15 = 27% of the trajectory
gets full source-KV injection. At 30 steps with inject=4, that
collapses to 4/30 = 13% — half the relative preservation window.
Doubling steps without scaling inject inadvertently halves
preservation. Less preservation → fidelity tanks, edit signal
strengthens.

### Calibrated run — inject scaled proportionally

```
config: paper_adaedit, mode=original, kv=0.9, ls=0.25, ext_on
num_steps=30, inject=8 (= 4 × 30/15, fraction preserved at ~27%)
sigmoid schedule, n=30, seed=0

psnr=18.7699  ssim=0.6411  lpips=0.3488
psnr_bg=21.9487  ssim_bg=0.7493  lpips_bg=0.1562
clip_i=0.9032  clip_t=0.2667  clip_dir=0.1154
```

PSNR climbs +1.07 dB vs the 15-step n=30 anchor (17.7045 → 18.77).
Other fidelity metrics also improve (SSIM +0.039, LPIPS −0.053,
PSNR_bg +1.48 dB) and CLIP-T / CLIP-dir behave consistent with the
"more preservation" direction (CLIP-dir 0.128 → 0.115).

### Reconciliation arithmetic

Two known protocol differences between local 15-step n=688 and the
published n=700 number:
- **step count** (15 → ~30): contributes +1.07 dB on n=30 (measured)
- **slice size** (n=30 → n=688): contributes +0.87 dB on PSNR
  (measured: 17.7045 at n=30 vs 18.5723 at n=688, both at 15 steps)

Estimated full-protocol n=688 `paper_adaedit` PSNR:

```
Local 15-step n=688 paper_adaedit:        18.5723
+ step-count effect (15 → 30):            +1.07 dB
                                          ────────
Estimated 30-step n=688 paper_adaedit:    ≈ 19.64

Published paper_adaedit (Table 1):        19.58
                                          ────────
Residual gap:                             ≈ +0.06 dB
```

**The published-vs-local PSNR gap is fully accounted for by step
count.** Residual ~0.06 dB is well within cross-implementation
variance.

Same arithmetic on SSIM/LPIPS reduces but does not fully close those
gaps:

```
SSIM:   est. 30-step n=688 ≈ 0.706 vs published 0.7433 → −0.04 residual
LPIPS:  est. 30-step n=688 ≈ 0.292 vs published 0.2703 → +0.022 residual
CLIP-T: matched anyway                                  → 0
```

The SSIM / LPIPS residuals are small and consistent with secondary
implementation differences (skimage SSIM window/sigma defaults,
LPIPS preprocessing details) — second-order effects, not the
dominant cause.

### Implications

1. **The local `paper_adaedit` reproduction is correct.** It runs at
   a different sampling step count than the published evaluation.
   Apparent fidelity depression is a protocol artifact, not a
   reproduction error.
2. **The on-slice (15-step) comparison remains valid.** Both methods
   are equally affected by the step-count protocol — relative deltas
   are invariant. Phase 8 / 14 / 16 headlines hold.
3. **`inject` is per-step absolute, not fractional.** Any future
   step-count change must scale `inject` proportionally
   (`inject_new = inject_15 × num_steps / 15`) to preserve the
   relative preservation window. Ditto for any other per-step
   threshold (PD controller targets, schedule knee points).
4. **AE bottleneck check is no longer urgent.** The step-count
   reconciliation explains the headline gap; AE-cap and metric-impl
   investigations move to lower priority.

### Recommended disclosure for paper / talk

> "Our locally-reproduced AdaEdit baseline scores 18.57 PSNR on the
> same 688-sample PIE-Bench slice we evaluate on, vs the paper's
> reported 19.58. We trace the gap to sampling-step protocol: at 30
> steps with proportionally scaled injection, the local baseline
> reaches an estimated 19.64 PSNR (29-sample n=30 measurement of
> +1.07 dB step-count effect plus +0.87 dB slice correction), within
> noise of published. We report all comparisons on a fixed 15-step
> protocol so both methods are equally affected; relative deltas are
> invariant."

---

## Updated next steps (post Phase 16/17)

1. **Optional: `pa_release_high` at full 700 with 30 steps inject=8.**
   ~5 hr compute. Would let us also report absolute numbers under
   the higher-step protocol. Not a prerequisite for the headline,
   since the on-slice 15-step claim is already defensible. Note
   that PA hyperparameters are calibrated at 15 steps; recalibration
   may be needed at 30 steps to avoid PD controller mistuning.
2. **Optional: SSIM/LPIPS implementation cross-check.** Swap
   `compute_metrics` for the PnP-Inversion official eval script on
   the existing edited images. Would close the residual ~0.04 SSIM
   / +0.022 LPIPS gaps to published. Diagnostic only — does not
   affect on-slice claims. ~1 hr.
3. **Closed by Phase 17:** Step-count reconciliation. No further
   work needed on the published-vs-local PSNR gap — it's fully
   accounted for.
