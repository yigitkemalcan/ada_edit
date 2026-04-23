import math
from typing import Callable
from copy import deepcopy

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

# AdaEdit exports
__all__ = [
    'get_noise', 'prepare', 'get_schedule', 'unpack',
    'denoise', 'denoise_rf_solver', 'denoise_fireflow', 'edit_uniedit',
    'latents_shift', 'channel_selective_latents_shift',
    'get_progressive_inject_schedule',
]

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0
):
    num_steps = len(timesteps[:-1])
    schedule_type = info.get('inject_schedule', 'binary')
    inject_weights = get_progressive_inject_schedule(num_steps, info['inject_step'], schedule_type)

    if inverse:
        timesteps = timesteps[::-1]
        inject_weights = inject_weights[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_weights[i] > 0.05
        info['inject_weight'] = inject_weights[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img = img + (t_prev - t_curr) * pred

    return img, info


def denoise_rf_solver(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0
):
    num_steps = len(timesteps[:-1])
    schedule_type = info.get('inject_schedule', 'binary')
    inject_weights = get_progressive_inject_schedule(num_steps, info['inject_step'], schedule_type)

    if inverse:
        timesteps = timesteps[::-1]
        inject_weights = inject_weights[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_weights[i] > 0.05
        info['inject_weight'] = inject_weights[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0
):
    num_steps = len(timesteps[:-1])
    schedule_type = info.get('inject_schedule', 'binary')
    inject_weights = get_progressive_inject_schedule(num_steps, info['inject_step'], schedule_type)

    if inverse:
        timesteps = timesteps[::-1]
        inject_weights = inject_weights[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    record_traj = bool(info.get('record_source_trajectory', False))
    traj: list = [] if record_traj else []
    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        if record_traj:
            traj.append(img.detach().clone())
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_weights[i] > 0.05
        info['inject_weight'] = inject_weights[i]

        if next_step_velocity is None:
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info
            )
        else:
            pred = next_step_velocity

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )
        next_step_velocity = pred_mid

        img = img + (t_prev - t_curr) * pred_mid

    if record_traj:
        traj.append(img.detach().clone())
        # Inversion runs timesteps reversed (clean -> noisy). Reverse so
        # traj[i] is the latent at the noise level matching target-pass
        # step i (target-pass start-of-step i sees t = timesteps[i]).
        if inverse:
            traj = traj[::-1]
        info['source_trajectory'] = traj

    return img, info


def edit_uniedit(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    src_txt=None,
    src_txt_ids=None,
    src_vec=None,
):
    inject_true = info['inject_step'] + int(info['alpha'] * len(timesteps[:-1]))
    schedule_type = info.get('inject_schedule', 'binary')
    inject_weights = get_progressive_inject_schedule(len(timesteps[:-1]), inject_true, schedule_type)

    if inverse:
        timesteps = timesteps[::-1]
        inject_weights = inject_weights[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    next_step_velocity = None
    step_threshold = round(info['alpha'] * len(timesteps[:-1]))
    
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        
        if inverse:
            if i >= step_threshold:
                continue 
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            info['t'] = t_prev if inverse else t_curr
            info['inverse'] = inverse
            info['second_order'] = False
            info['inject'] = inject_weights[i] > 0.05
            info['inject_weight'] = inject_weights[i]

            if info['kv_mask']:
                pred_source,info = model(
                    img=img,
                    img_ids=img_ids,
                    txt=src_txt,
                    txt_ids=src_txt_ids,
                    y=src_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    info=info
                )
            if next_step_velocity is None:
                pred, info = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    info=info
                )
            else:
                pred = next_step_velocity
            
            img_next = img + (t_prev - t_curr) * pred

            t_vec_next = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
            info['second_order'] = True
            pred_next, info = model(
                img=img_next,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_next,
                guidance=guidance_vec,
                info=info
            )
            next_step_velocity = pred_next
            
            img = img + (t_prev - t_curr) * pred_next
            
        else:
            if i < (len(timesteps[:-1]) - step_threshold):
                continue 
            
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            info['t'] = t_prev if inverse else t_curr
            info['inverse'] = inverse
            info['second_order'] = False
            info['inject'] = inject_weights[i] > 0.05
            info['inject_weight'] = inject_weights[i]


            src_info = deepcopy(info)
            
            info['kv_mix'] = False
            pred_src, info = model(
                img=img,
                img_ids=img_ids,
                txt=src_txt,
                txt_ids=src_txt_ids,
                y=src_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info
            )

            info['kv_mix'] = True
            pred_trg, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info
            )
            
            
            cfg_component = pred_trg - pred_src
            save_sub_map = cfg_component.abs().mean(dim=2, keepdim=True)        # [b, h*w, c] TODO 暂只支持batch_size=1
            save_sub_map = (save_sub_map - save_sub_map.min()) / (save_sub_map.max() - save_sub_map.min())
            
            # velocity fusion
            fused_v = save_sub_map * pred_trg + (1 - save_sub_map) * pred_src
            # result
            pred = fused_v + (save_sub_map + 1) * info['omega'] * cfg_component
            
            img = img + (t_prev - t_curr) * pred

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def compute_mean_std(x: torch.Tensor, eps: float = 1e-5):
    if x.dim() == 4:  # [B, C, H, W]
        dims = (2, 3)
    elif x.dim() == 3:  # [B, L, (C ph pw)]
        dims = (1,)
    else:
        raise ValueError(f"Unsupported tensor dim: {x.shape}")

    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    return mean, std

def latents_shift(x_content: torch.Tensor, x_style: torch.Tensor, indices: torch.Tensor, alpha: float = 0.25, eps: float = 1e-5):

    out = x_content.clone()

    c_sel = x_content[:, indices, :]   # [B, K, C]
    s_sel = x_style[:, indices, :]     # [B, K, C]

    c_mean, c_std = compute_mean_std(c_sel, eps)  # [B, 1, C]
    s_mean, s_std = compute_mean_std(s_sel, eps)  # [B, 1, C]

    normalized = (c_sel - c_mean) / c_std
    stylized   = normalized * s_std + s_mean

    stylized = alpha * stylized + (1 - alpha) * c_sel

    out[:, indices, :] = stylized
    return out


# ============================================================
# AdaEdit: Channel-Selective Latents-Shift
# ============================================================

def channel_selective_latents_shift(
    x_content: torch.Tensor,
    x_style: torch.Tensor,
    indices: torch.Tensor,
    alpha: float = 0.25,
    temperature: float = 1.0,
    eps: float = 1e-5,
):
    """
    Channel-selective Latents-Shift: different channels get different shift strengths.
    Channels with larger source-random distribution gap are more edit-related and get stronger shift.
    """
    out = x_content.clone()

    c_sel = x_content[:, indices, :]   # [B, K, C]
    s_sel = x_style[:, indices, :]     # [B, K, C]

    # Per-channel importance: how different is source vs random in each channel
    c_chan_mean = c_sel.mean(dim=1)  # [B, C]
    s_chan_mean = s_sel.mean(dim=1)  # [B, C]
    channel_diff = (c_chan_mean - s_chan_mean).abs()  # [B, C]
    channel_weight = torch.softmax(channel_diff / temperature, dim=-1)  # [B, C]
    # Scale so mean weight = 1.0 (preserves overall alpha strength)
    channel_weight = channel_weight * channel_weight.shape[-1]  # [B, C]

    # Per-channel AdaIN
    c_mean = c_sel.mean(dim=1, keepdim=True)  # [B, 1, C]
    c_std = c_sel.var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt()
    s_mean = s_sel.mean(dim=1, keepdim=True)
    s_std = s_sel.var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt()

    normalized = (c_sel - c_mean) / c_std
    stylized = normalized * s_std + s_mean

    # Channel-adaptive alpha: alpha_c = alpha * w_c
    alpha_c = (alpha * channel_weight).unsqueeze(1)  # [B, 1, C]
    alpha_c = alpha_c.clamp(0.0, 1.0)

    shifted = alpha_c * stylized + (1 - alpha_c) * c_sel

    out[:, indices, :] = shifted
    return out


# ============================================================
# AdaEdit: Progressive Injection Schedule
# ============================================================

def get_progressive_inject_schedule(num_steps: int, inject_steps: int, schedule_type: str = 'cosine'):
    """
    Returns a list of continuous injection weights [0, 1] instead of binary [True/False].
    schedule_type: 'cosine', 'sigmoid', 'linear', 'binary' (original ProEdit)
    """
    if schedule_type == 'binary':
        return [1.0] * inject_steps + [0.0] * (num_steps - inject_steps)

    weights = []
    for i in range(num_steps):
        if inject_steps <= 0:
            weights.append(0.0)
            continue
        ratio = i / max(inject_steps - 1, 1)
        if schedule_type == 'cosine':
            w = 0.5 * (1 + math.cos(math.pi * min(ratio, 1.0)))
        elif schedule_type == 'sigmoid':
            k = 5.0  # moderate sharpness
            midpoint = 0.7  # shift midpoint later to keep more steps active
            w = 1.0 / (1.0 + math.exp(k * (ratio - midpoint)))
        elif schedule_type == 'linear':
            w = max(1.0 - ratio, 0.0)
        else:
            w = 1.0 if i < inject_steps else 0.0
        weights.append(w)
    return weights