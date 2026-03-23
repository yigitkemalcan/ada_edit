<h3 align="center">
    AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing
</h3>

<p align="center">
<a href="https://arxiv.org/abs/xxxx.xxxxx"><img alt="Paper" src="https://img.shields.io/badge/Paper-AdaEdit-b31b1b.svg"></a>
<a href="https://github.com/leeguandong/AdaEdit"><img src="https://img.shields.io/static/v1?label=GitHub&message=repository&color=green"></a>
</p>

<p align="center">
<span style="color:#137cf3; font-family: Gill Sans">Guandong Li,</span><sup></sup></a>
<span style="color:#137cf3; font-family: Gill Sans">Zhaobin Chu</span></a> <br>
<span style="font-size: 13.5px">iFLYTEK Research</span><br>
</p>

## Abstract

**AdaEdit** is a **training-free** adaptive image editing framework designed for Flow-based diffusion models. We address the "injection dilemma" in inversion-based editing through two core innovations:

**(1) Progressive Injection Schedule**: Replaces binary truncation with continuous decay functions (sigmoid/cosine/linear), eliminating feature discontinuities and reducing hyperparameter sensitivity.

**(2) Channel-Selective Latent Perturbation**: Applies strong perturbations to edit-relevant channels while maintaining weak perturbations on structural channels based on channel importance estimation, achieving effective editing while preserving structural fidelity.

On PIE-Bench (700 images), AdaEdit achieves significant improvements over baseline methods:
- **LPIPS ↓ 8.7%** (background preservation)
- **SSIM ↑ 2.6%** (structural similarity)
- **PSNR ↑ 2.3%** (peak signal-to-noise ratio)
- **CLIP ≈ -0.9%** (editing accuracy nearly lossless)

## Installation

```bash
# Clone repository
git clone https://github.com/leeguandong/AdaEdit.git
cd AdaEdit

# Install dependencies
pip install -r requirements.txt
```

## Model Download

Download FLUX.1-dev model weights and place them in the `checkpoints/` directory:

```bash
# Download FLUX.1-dev model
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/flux-dev
```

## Usage

### Basic Usage

```bash
python adaedit.py \
    --source_img source.jpg \
    --source_prompt "A photo of a cat" \
    --target_prompt "A photo of a dog" \
    --output_dir outputs/
```

### Full Parameters

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A photo of a cat" \
    -tp "A photo of a dog" \
    -o outputs/ \
    --edit_object "cat" \
    --num_steps 15 \
    --guidance 4.0 \
    --inject 4 \
    --inject_schedule sigmoid \
    --kv_mix_ratio 0.9 \
    --ls_ratio 0.25 \
    --use_channel_ls \
    --channel_ls_temp 1.0 \
    --seed 42
```

### Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--source_img` | `-i` | **Required** | Source image path |
| `--source_prompt` | `-sp` | **Required** | Source image description |
| `--target_prompt` | `-tp` | **Required** | Target editing description |
| `--output_dir` | `-o` | `outputs/` | Output directory |
| `--edit_object` | | `""` | Object to edit (for mask extraction) |
| `--num_steps` | | `15` | Number of sampling steps |
| `--guidance` | | `4.0` | Guidance scale |
| `--inject` | | `4` | Injection step threshold |
| `--inject_schedule` | | `sigmoid` | Injection schedule: `binary`/`sigmoid`/`cosine`/`linear` |
| `--kv_mix_ratio` | | `0.9` | KV-Mix ratio |
| `--ls_ratio` | | `0.25` | Latents-Shift strength |
| `--use_channel_ls` | | `False` | Enable channel-selective Latents-Shift |
| `--channel_ls_temp` | | `1.0` | Channel importance temperature |
| `--seed` | | `0` | Random seed (0=random) |
| `--offload` | | `False` | Low VRAM mode |

### Examples

```bash
# Example 1: Object replacement (with progressive sigmoid schedule)
python adaedit.py \
    -i examples/cat.jpg \
    -sp "A photo of a cat on a sofa" \
    -tp "A photo of a dog on a sofa" \
    --edit_object "cat" \
    --inject_schedule sigmoid \
    --use_channel_ls

# Example 2: Style transfer (without Latents-Shift)
python adaedit.py \
    -i examples/portrait.jpg \
    -sp "A portrait photo" \
    -tp "A portrait in anime style" \
    --edit_type style \
    --inject_schedule cosine

# Example 3: Low VRAM mode
python adaedit.py \
    -i examples/scene.jpg \
    -sp "A street scene" \
    -tp "A street scene at night" \
    --offload
```

## Key Features

### 1. Progressive Injection Schedule

Traditional methods use binary truncation (first N steps inject=1, subsequent steps inject=0), causing feature discontinuities. AdaEdit provides three continuous decay functions:

- **Sigmoid** (recommended): Smooth transition, moderate sharpness
- **Cosine**: Cosine decay, smoother
- **Linear**: Linear decay, simplest

### 2. Channel-Selective Latent Perturbation

Different channels encode different information (structure/color/texture). AdaEdit automatically estimates the edit-relevance of each channel:
- Edit-relevant channels: Strong perturbation (promotes content change)
- Structural channels: Weak perturbation (maintains layout stability)

### 3. Plug-and-Play

AdaEdit is training-free and works with multiple ODE solvers:
- Euler (basic)
- RF-Solver (second-order)
- FireFlow (velocity reuse, recommended)

## Python API

```python
from api import AdaEditPipeline

# Initialize
pipeline = AdaEditPipeline(
    model_path="checkpoints/flux-dev",
    device="cuda"
)

# Edit image
result = pipeline.edit(
    source_image="source.jpg",
    source_prompt="A photo of a cat",
    target_prompt="A photo of a dog",
    edit_object="cat",
    inject_schedule="sigmoid",
    use_channel_ls=True,
    num_steps=15,
    guidance=4.0,
    seed=42
)

# Save result
result.save("output.jpg")
```

## Technical Details

### Progressive Injection Weight Calculation

```python
# Sigmoid schedule
w(t) = 1 / (1 + exp(k * (t/T_inj - 0.7)))

# Cosine schedule
w(t) = 0.5 * (1 + cos(π * t/T_inj))

# Linear schedule
w(t) = max(1 - t/T_inj, 0)
```

### Channel Importance Estimation

```python
# Calculate distribution difference for each channel
channel_diff = |mean(source_channel) - mean(random_channel)|

# Softmax normalization
channel_weight = softmax(channel_diff / temperature) * num_channels
```

## Citation

If you use AdaEdit in your research, please cite our paper:

```bibtex
@article{li2026adaedit,
  title={AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing},
  author={Li, Guandong and Chu, Zhaobin},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## Acknowledgments

- [FLUX.1](https://github.com/black-forest-labs/flux) - Base flow model
- [FireFlow](https://github.com/xxx/fireflow) - Efficient ODE solver
- [PIE-Bench](https://github.com/xxx/pie-bench) - Evaluation benchmark

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
