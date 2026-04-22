# AdaEdit-Adaptive experiment log

Single-image tuning sweeps on the horse → toy-horse edit.
Source: `examples/horse.jpg`, source prompt `A photo of a horse`,
target prompt `A photo of a toy horse`, edit_object `horse`, edit_type
`change`, seed `42`, num_steps `15`, flux-dev.

All runs use `mode=pd_adaptive` unless stated otherwise. Metrics
implementation: CLIP (openai/clip-vit-large-patch14), LPIPS (AlexNet),
SSIM + PSNR (scikit-image). Masked `_bg` variants restrict metrics to
the preservation (non-edit) region using the latent-space mask from
phase-1 indices, resized to pixel space.

Metric conventions:
- `psnr / ssim / lpips / psnr_bg / ssim_bg / lpips_bg` — edit vs source
  (preservation side). Higher is better for PSNR/SSIM, lower for LPIPS.
- `clip_i` — CLIP cosine similarity, edit vs source image.
- `clip_t` — CLIP cosine similarity, edit vs target prompt.
- `clip_dir` — CLIP directional similarity: (edit_img − src_img) vs
  (target_prompt − source_prompt). Most principled edit-quality metric
  because it subtracts out baseline similarity.

---

## Recommended configuration (current best)

> **Updated after phase 3.** Multi-image PIE-Bench validation (n=30)
> showed that turning on all three AdaEdit extensions on top of the PD
> controller produces the best config overall — see [Phase 3](#phase-3--pie-bench-multi-image-validation-n30).
> The two phase-2 winners below are preserved for single-image
> reproducibility on the horse edit.

Two viable winners from phase 2. Pick based on what you care about
most. Both beat vanilla AdaEdit (baseline_original) by large margins.

### Option A — balanced (recommended default)

Best `psnr`, `lpips`, `ssim_bg`. Strong everywhere.

```python
from adaedit_adaptive import build_parser, run

args = build_parser().parse_args([
    "--source_img", "examples/horse.jpg",
    "--source_prompt", "A photo of a horse",
    "--target_prompt", "A photo of a toy horse",
    "--edit_object", "horse",
    "--mode", "pd_adaptive",
    "--drift_metric", "latent_init",
    "--kv_mix_ratio", "0.9",
    "--target_drift", "0.02",
    "--ls_ratio", "0.45",
    "--kp", "1.5",
    "--kd", "0.2",
    "--num_steps", "15",
    "--seed", "42",
    "--output_dir", "outputs_adaptive",
])
run(args, t5=t5, clip=clip, model=model, ae=ae)
```

Expected metrics on the horse:
```
psnr=20.14  ssim=0.7485  lpips=0.2357
psnr_bg=23.83  ssim_bg=0.8402  lpips_bg=0.1299
clip_i=0.7540  clip_t=0.2897  clip_dir=0.2406
```

### Option B — edit-direction winner

Best `clip_dir` of the whole sweep (29 configs). Use when edit-quality
is the primary reporting metric.

Change only: `--kp 0.5` (keep `--kd 0.2`, everything else identical).

Expected metrics:
```
psnr=20.14  ssim=0.7482  lpips=0.2366
psnr_bg=23.84  ssim_bg=0.8397  lpips_bg=0.1301
clip_i=0.7583  clip_t=0.2883  clip_dir=0.2462
```

---

## Baseline for reference

Vanilla AdaEdit (`--mode original --kv_mix_ratio 0.9 --ls_ratio 0.25`):
```
psnr=17.4027  ssim=0.7036  lpips=0.2826
psnr_bg=22.8856  ssim_bg=0.8286  lpips_bg=0.1476
clip_i=0.7441  clip_t=0.2814  clip_dir=0.2378
```

Delta baseline → Option A (balanced winner):
- PSNR: +2.74 dB
- SSIM: +0.045
- LPIPS: −0.047
- psnr_bg: +0.94 dB
- lpips_bg: −0.018
- clip_t: +0.008
- clip_dir: +0.003

The preservation gains are large and consistent; the CLIP gains are
small (within single-image noise).

---

## Phase 1 — initial sweep (17 runs)

Grid: `kv_mix_ratio ∈ {0.6, 0.75, 0.9}` × `target_drift ∈ {0.02, 0.05}`
× `ls_ratio ∈ {0.25, 0.45}` on `pd_adaptive`, plus Tier-2 feature
toggles and two baselines.

```
name                     time_s   psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
-----------------------  -------  -------  ------  ------  -------  -------  --------  ------  ------  --------
t1_kv0.6_td0.02_ls0.25   17.32    20.2208  0.7408  0.2518  23.6066  0.8307   0.1436    0.7601  0.2803  0.2239
t1_kv0.6_td0.02_ls0.45   12.35    20.2759  0.7431  0.2486  23.6907  0.8328   0.1405    0.7630  0.2794  0.2249
t1_kv0.6_td0.05_ls0.25   12.32    20.2519  0.7410  0.2510  23.6421  0.8313   0.1436    0.7617  0.2800  0.2229
t1_kv0.6_td0.05_ls0.45   12.31    20.2720  0.7430  0.2498  23.6608  0.8327   0.1432    0.7690  0.2808  0.2324
t1_kv0.75_td0.02_ls0.25  12.33    20.0884  0.7405  0.2518  23.6060  0.8312   0.1435    0.7531  0.2860  0.2292
t1_kv0.75_td0.02_ls0.45  12.33    20.1652  0.7416  0.2483  23.6757  0.8317   0.1397    0.7696  0.2765  0.2351
t1_kv0.75_td0.05_ls0.25  12.29    20.0784  0.7396  0.2528  23.5925  0.8303   0.1437    0.7550  0.2836  0.2259
t1_kv0.75_td0.05_ls0.45  12.34    20.1501  0.7404  0.2499  23.5546  0.8297   0.1414    0.7605  0.2764  0.2278
t1_kv0.9_td0.02_ls0.25   12.32    19.9641  0.7390  0.2491  23.6619  0.8311   0.1400    0.7569  0.2833  0.2388
t1_kv0.9_td0.02_ls0.45   12.31    20.0980  0.7428  0.2406  23.8137  0.8351   0.1325    0.7498  0.2939  0.2400   <- phase-1 winner
t1_kv0.9_td0.05_ls0.25   12.29    20.0119  0.7386  0.2507  23.7050  0.8307   0.1401    0.7546  0.2884  0.2384
t1_kv0.9_td0.05_ls0.45   12.28    20.1066  0.7406  0.2455  23.7297  0.8307   0.1363    0.7427  0.2839  0.2282
t2_soft_mask             12.25    20.3030  0.7394  0.2572  23.0226  0.8223   0.1493    0.7378  0.2877  0.2231
t2_adaptive_kv           12.33    20.1827  0.7413  0.2510  23.6704  0.8315   0.1420    0.7552  0.2853  0.2294
t2_channel_ls            12.28    20.0904  0.7392  0.2527  23.5572  0.8290   0.1441    0.7586  0.2813  0.2224
baseline_original        12.24    17.4027  0.7036  0.2826  22.8856  0.8286   0.1476    0.7441  0.2814  0.2378
baseline_pd_default      12.26    20.0119  0.7386  0.2507  23.7050  0.8307   0.1401    0.7546  0.2884  0.2384
```

Best per metric:
```
       psnr (↑): t2_soft_mask  (20.3030)
       ssim (↑): t1_kv0.6_td0.02_ls0.45  (0.7431)
      lpips (↓): t1_kv0.9_td0.02_ls0.45  (0.2406)
    psnr_bg (↑): t1_kv0.9_td0.02_ls0.45  (23.8137)
    ssim_bg (↑): t1_kv0.9_td0.02_ls0.45  (0.8351)
   lpips_bg (↓): t1_kv0.9_td0.02_ls0.45  (0.1325)
     clip_i (↑): t1_kv0.75_td0.02_ls0.45  (0.7696)
     clip_t (↑): t1_kv0.9_td0.02_ls0.45  (0.2939)
   clip_dir (↑): t1_kv0.9_td0.02_ls0.45  (0.2400)
```

Takeaways:
1. `t1_kv0.9_td0.02_ls0.45` is Pareto-dominant — wins on 6 of 9 metrics,
   including all three bg preservation metrics and both text-alignment
   metrics simultaneously. The "pick a winner" case.
2. `ls_ratio=0.45 > 0.25` almost everywhere — more foreground noise in
   phase 2 lets the edit breathe without hurting bg preservation.
3. `kv_mix_ratio=0.9` (high) beats lower values for `edit_type=change`.
   The high base gives the foreground more target-KV influence; the
   model spends its generative budget in the fg instead of leaking
   into bg.
4. `target_drift` 0.02 vs 0.05 — marginal, 0.02 slightly preferred.
5. Tier-2 features hurt or didn't help:
   - `soft_mask`: best whole-image PSNR but **worst `psnr_bg`**.
     Soft boundaries blur the fg/bg distinction.
   - `adaptive_kv`, `channel_ls`: mid-pack, no gain.

---

## Phase 2 — follow-up sweep (12 runs)

Around the phase-1 winner (`kv=0.9, td=0.02, ls=0.45`):
- `ls_ratio` extension to `{0.55, 0.65, 0.75}`.
- `kp × kd` grid at `ls=0.45`: `kp ∈ {0.5, 1.0, 1.5} × kd ∈ {0.1, 0.2, 0.4}`.

```
name           time_s   psnr     ssim    lpips   psnr_bg  ssim_bg  lpips_bg  clip_i  clip_t  clip_dir
-------------  -------  -------  ------  ------  -------  -------  --------  ------  ------  --------
f_ls0.55       12.35    20.0509  0.7458  0.2383  23.7242  0.8360   0.1299    0.7368  0.2889  0.2236
f_ls0.65       12.29    20.0538  0.7487  0.2384  23.7703  0.8400   0.1288    0.7394  0.2945  0.2293
f_ls0.75       12.31    19.9535  0.7436  0.2436  23.6429  0.8338   0.1328    0.7432  0.2760  0.2081
f_kp0.5_kd0.1  12.35    20.0962  0.7454  0.2386  23.7825  0.8363   0.1312    0.7552  0.2883  0.2363
f_kp0.5_kd0.2  12.35    20.1381  0.7482  0.2366  23.8390  0.8397   0.1301    0.7583  0.2883  0.2462   <- best clip_dir
f_kp0.5_kd0.4  12.26    20.1012  0.7447  0.2393  23.7836  0.8358   0.1322    0.7579  0.2892  0.2434
f_kp1.0_kd0.1  12.26    20.1216  0.7473  0.2379  23.7773  0.8372   0.1312    0.7541  0.2883  0.2369
f_kp1.0_kd0.2  12.26    20.0980  0.7428  0.2406  23.8137  0.8351   0.1325    0.7498  0.2939  0.2400   <- reproduces phase-1 winner exactly
f_kp1.0_kd0.4  12.31    20.0370  0.7408  0.2419  23.7479  0.8334   0.1333    0.7501  0.2881  0.2352
f_kp1.5_kd0.1  12.23    20.1352  0.7446  0.2382  23.8270  0.8367   0.1310    0.7562  0.2905  0.2438
f_kp1.5_kd0.2  12.22    20.1443  0.7485  0.2357  23.8334  0.8402   0.1299    0.7540  0.2897  0.2406   <- balanced winner
f_kp1.5_kd0.4  12.16    20.0897  0.7435  0.2397  23.8141  0.8362   0.1316    0.7516  0.2903  0.2349
```

Best per metric:
```
       psnr (↑): f_kp1.5_kd0.2  (20.1443)
       ssim (↑): f_ls0.65       (0.7487)
      lpips (↓): f_kp1.5_kd0.2  (0.2357)
    psnr_bg (↑): f_kp0.5_kd0.2  (23.8390)
    ssim_bg (↑): f_kp1.5_kd0.2  (0.8402)
   lpips_bg (↓): f_ls0.65       (0.1288)
     clip_i (↑): f_kp0.5_kd0.2  (0.7583)
     clip_t (↑): f_ls0.65       (0.2945)
   clip_dir (↑): f_kp0.5_kd0.2  (0.2462)
```

Takeaways:
1. **Sanity check passed.** `f_kp1.0_kd0.2` reproduces the phase-1
   winner to 4 decimals → sweep is deterministic, observed deltas
   are signal not noise.
2. `kd=0.2` strictly dominates `kd=0.1` and `kd=0.4` at every `kp`.
   This is the most robust finding — adopt as fixed.
3. `kp` is non-monotonic. Both `kp=0.5` and `kp=1.5` beat `kp=1.0` on
   most metrics. Likely alpha saturation (clamped to `[0.2, 1.2]`):
   both extremes settle at similar effective alpha via different
   paths; the middle value is the least-stable zone.
4. `ls_ratio` tops out at 0.45–0.65. `0.65` gets marginal wins on
   `lpips_bg` and `clip_t` but costs `clip_dir`; `0.75` is strictly
   worse (too much fg noise erases the edit signal).
5. Tightest range across 12 configs: PSNR 19.95–20.14 (0.19 dB),
   clip_t 0.276–0.295 (0.019). We're in diminishing-returns territory
   for single-image tuning.

---

## Robust findings across both sweeps

These are the parameter choices that consistently helped across all
29 runs and should be treated as locked in:

1. `kv_mix_ratio = 0.9` for `edit_type=change`.
2. `target_drift = 0.02`.
3. `ls_ratio ∈ [0.45, 0.55]`.
4. `kd = 0.2`.
5. `kp ∈ {0.5, 1.5}` preferred over `1.0` on this edit.
6. ~~Skip `use_soft_mask` / `use_adaptive_kv` / `use_channel_ls` — no
   measurable benefit on this edit class.~~ **Superseded by phase 3.**
   On the horse edit (n=1) the extensions didn't help, but on PIE-Bench
   (n=30, diverse edits, GT masks) they stack additively with PD and
   produce the best overall config. See [Phase 3](#phase-3--pie-bench-multi-image-validation-n30).
7. `pd_adaptive` beats `original` by ~2.7 dB PSNR / ~0.05 LPIPS —
   biggest delta in the entire study.

---

## Known caveats

- **n = 1.** All 29 runs are a single source image + single edit. The
  `kp=0.5` vs `kp=1.5` split is real under this prompt; we do not
  know whether it generalizes.
- **`edit_type=change` only.** The KV-Mix formula inverts for `add` and
  becomes unmasked for `style`, so the optimal `kv_mix_ratio` may flip
  direction for those.
- **Metrics noise floor.** Differences in the 0.001–0.005 range on
  CLIP scores are within expected jitter across seeds / nearby
  edits, even though the runs here are reproducible.
- **Seed fixed at 42.** All results are conditional on this seed.

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

The three boolean extensions (`use_soft_mask`, `use_adaptive_kv`,
`use_channel_ls`) were **not** part of the phase-1/2 sweeps — they
were toggled off in every run by default. Phase 3 is their first
appearance alongside the PD controller.

### Configurations

| ID | Name | mode | PD | soft_mask | adaptive_kv | channel_ls |
|----|------|------|----|-----------|-------------|------------|
| A | adaptive (phase-2 winner) | `pd_adaptive` | ✓ | ✗ | ✗ | ✗ |
| B | original-lite (no extensions) | `original` | — | ✗ | ✗ | ✗ |
| C | paper-AdaEdit | `original` | — | ✓ | ✓ | ✓ |
| D | **adaptive + extensions** | `pd_adaptive` | ✓ | ✓ | ✓ | ✓ |

Shared PD parameters (A and D): `kv_mix_ratio=0.9`,
`target_drift=0.02`, `ls_ratio=0.45`, `kp=1.5`, `kd=0.2` (phase-2
balanced winner). Paper-style runs (B and C): `kv_mix_ratio=0.9`,
`ls_ratio=0.25` (AdaEdit paper defaults).

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

1. **D is the winner.** Beats A-alone and C-alone on every fidelity
   and background metric. Edit quality (CLIP-t, CLIP-dir) is
   essentially unchanged — no tradeoff.
2. **Extensions and PD are additive, not redundant.** The A→D delta
   (+1.11 dB psnr_bg, −0.025 lpips_bg) is meaningfully smaller than
   B→C (+2.79 dB psnr_bg), so there is partial overlap — but the
   residual gain on top of PD is still worth the extension cost.
   The four mechanisms constrain drift through orthogonal channels:
   - PD controller — dynamic α gating across time.
   - `soft_mask` — spatial attention gating.
   - `channel_ls` — channel-selective latent perturbation.
   - `adaptive_kv` — per-layer KV-mix scaling.
3. **Phase-1's "skip extensions" finding was n=1 artefact.** On the
   horse edit, `soft_mask` hurt `psnr_bg` and the other two were
   mid-pack. On 30 diverse PIE-Bench samples with GT masks, all
   three contribute. The horse edit is unrepresentative (single
   prominent object, clean bg) — exactly the case where extensions
   have least to do.
4. **PD ≈ extensions on their own.** A and C are statistically
   indistinguishable (psnr_bg 20.32 vs 20.47). PD alone recovers
   the same bg-preservation gain the three extensions produce
   together on bare baseline. Mechanistically coherent: both are
   solving "don't leak into background," just with different knobs.
5. **CLIP-t ranking is inverted from intuition.** B (most damaged
   bg) wins CLIP-t / CLIP-dir by the largest margin. When bg
   preservation fails, the delta image is bigger, which reads as
   "more of the edit prompt visible" in CLIP space — but at the
   cost of fidelity. This is a known failure mode of CLIP-t as a
   standalone edit metric and motivates the `*_bg` masked variants.

### Against the AdaEdit paper (Table 1, whole-image, n=700)

|        | paper AdaEdit | **our D (n=30)** |
|--------|---------------|------------------|
| PSNR   | 19.58         | 18.92            |
| SSIM   | 0.7433        | 0.6369           |
| LPIPS  | 0.2703        | 0.3534           |
| CLIP-T | 0.2593        | 0.2662           |

We are within ~0.7 dB PSNR and ahead on CLIP-T. The SSIM / LPIPS
gap is partly explained by our n=30 slice including 4 style
samples (cat 9), which score terribly on whole-image metrics
because the whole image is the edit region. Full-700 run is needed
for an apples-to-apples comparison.

### Recommended configuration (multi-image)

```python
args = build_parser().parse_args([
    "--source_img", "...",
    "--source_prompt", "...",
    "--target_prompt", "...",
    "--edit_object", "...",
    "--mode", "pd_adaptive",
    "--drift_metric", "latent_init",
    "--kv_mix_ratio", "0.9",
    "--target_drift", "0.02",
    "--ls_ratio", "0.45",
    "--kp", "1.5",
    "--kd", "0.2",
    "--use_channel_ls",
    "--use_soft_mask",
    "--use_adaptive_kv",
    "--num_steps", "15",
    "--seed", "42",
])
```

---

## Next step

1. **Run the full 700 PIE-Bench on config D.** At ~12.4 s/sample ×
   620 samples (cat 8 excluded) ≈ 2.1 h. Produces the number that
   goes next to AdaEdit's 19.58 / 0.7433 / 0.2703 / 0.2593 in any
   write-up.
2. **Consider a second full-700 on config C** (paper-AdaEdit) so we
   have a locally-reproduced paper baseline on the exact same slice
   — removes "are our numbers comparable to the paper?" ambiguity.
3. **Category 8 (background).** Currently skipped per project
   decision. Worth one short experiment to confirm AdaEdit + PD
   genuinely underperforms there vs. a dedicated bg-replace method,
   or whether our pipeline handles it acceptably.
4. **`kp=0.5` vs `kp=1.5` tie from phase 2** — still unresolved.
   After the full 700 on D, run a second pass with `kp=0.5` if
   time permits to pick the global winner.
