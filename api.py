"""
AdaEdit Python API
Provides a simple interface for programmatic image editing.
"""
import torch
import numpy as np
from PIL import Image
from typing import Optional, Literal

from src.flux.sampling import (
    denoise_fireflow,
    get_schedule,
    prepare,
    unpack,
    get_noise,
    latents_shift,
    channel_selective_latents_shift,
)
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    get_word_index,
)


class AdaEditPipeline:
    """AdaEdit image editing pipeline."""

    def __init__(
        self,
        model_path: str = "checkpoints/flux-dev",
        model_name: str = "flux-dev",
        device: str = "cuda",
        offload: bool = False,
    ):
        """
        Initialize AdaEdit pipeline.

        Args:
            model_path: Path to model checkpoints
            model_name: Model name (flux-dev or flux-schnell)
            device: Device to run on (cuda or cpu)
            offload: Enable model offloading for low VRAM
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.offload = offload

        print("Loading models...")
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)

        if offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        print("✓ Models loaded successfully")

    @torch.inference_mode()
    def edit(
        self,
        source_image: str | np.ndarray | Image.Image,
        source_prompt: str,
        target_prompt: str,
        edit_object: str = "",
        edit_type: Literal["add", "remove", "change", "style"] = "change",
        num_steps: int = 15,
        guidance: float = 4.0,
        inject: int = 4,
        inject_schedule: Literal["binary", "sigmoid", "cosine", "linear"] = "sigmoid",
        kv_mix_ratio: float = 0.9,
        ls_ratio: float = 0.25,
        use_channel_ls: bool = True,
        channel_ls_temp: float = 1.0,
        use_soft_mask: bool = False,
        soft_mask_gamma: float = 5.0,
        use_adaptive_kv: bool = False,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Edit an image using AdaEdit.

        Args:
            source_image: Source image (path, numpy array, or PIL Image)
            source_prompt: Description of source image
            target_prompt: Target editing prompt
            edit_object: Object to edit (for mask extraction)
            edit_type: Type of edit (add/remove/change/style)
            num_steps: Number of sampling steps
            guidance: Guidance scale
            inject: Injection step threshold
            inject_schedule: Injection schedule type
            kv_mix_ratio: KV-Mix ratio
            ls_ratio: Latents-Shift strength
            use_channel_ls: Use channel-selective Latents-Shift
            channel_ls_temp: Temperature for channel importance
            use_soft_mask: Use soft mask
            soft_mask_gamma: Soft mask sharpness
            use_adaptive_kv: Use layer-adaptive KV-Mix
            seed: Random seed (None for random)

        Returns:
            Edited PIL Image
        """
        torch.set_grad_enabled(False)

        # Load and preprocess image
        if isinstance(source_image, str):
            img = np.array(Image.open(source_image).convert('RGB'))
        elif isinstance(source_image, Image.Image):
            img = np.array(source_image.convert('RGB'))
        else:
            img = source_image

        h, w = img.shape[:2]
        new_h = h if h % 16 == 0 else h - h % 16
        new_w = w if w % 16 == 0 else w - w % 16
        img = img[:new_h, :new_w, :]

        # Encode image
        if self.offload:
            self.ae.encoder.to(self.device)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        init_latent = self.ae.encode(img_tensor).to(torch.bfloat16)

        # Setup seed
        if seed is None:
            seed = torch.Generator(device="cpu").seed()

        # Prepare random noise
        z_random = get_noise(1, new_h, new_w, device=self.device,
                           dtype=torch.bfloat16, seed=seed)

        # Offload and prepare text embeddings
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        inp_source = prepare(self.t5, self.clip, init_latent, prompt=source_prompt)
        inp_target = prepare(self.t5, self.clip, init_latent, prompt=target_prompt)
        inp_random = prepare(self.t5, self.clip, z_random, prompt=target_prompt)

        # Get word index
        key_word_index = get_word_index(source_prompt, edit_object, self.t5)

        # Setup info dict
        info = {
            'feature_path': 'features',
            'feature': {},
            'inject_step': inject,
            'sampling_solver': 'fireflow',
            'latent_h': new_h // 16,
            'latent_w': new_w // 16,
            'key_word_index': key_word_index,
            'edit_type': edit_type,
            'kv_mix_ratio': kv_mix_ratio,
            'indices': [],
            'kv_mix': True,
            'kv_mask': True,
            'use_soft_mask': use_soft_mask,
            'soft_mask_gamma': soft_mask_gamma,
            'use_adaptive_kv': use_adaptive_kv,
            'use_channel_ls': use_channel_ls,
            'channel_ls_temp': channel_ls_temp,
            'inject_schedule': inject_schedule,
        }

        # Get timesteps
        timesteps = get_schedule(num_steps, inp_source["img"].shape[1],
                                shift=(self.model_name != "flux-schnell"))
        info['mask_time'] = timesteps[-2]

        # Offload and run inversion
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # Phase 1: Inversion
        z_inv, info = denoise_fireflow(self.model, **inp_source, timesteps=timesteps,
                                      guidance=1.0, inverse=True, info=info)

        # Phase 2: Latent perturbation
        if edit_type != 'style':
            if use_channel_ls:
                z_inv = channel_selective_latents_shift(
                    z_inv, inp_random["img"], info['indices'],
                    alpha=ls_ratio, temperature=channel_ls_temp
                )
            else:
                z_inv = latents_shift(z_inv, inp_random["img"],
                                     info['indices'], alpha=ls_ratio)

        inp_target["img"] = z_inv
        timesteps = get_schedule(num_steps, inp_target["img"].shape[1],
                                shift=(self.model_name != "flux-schnell"))

        # Phase 3: Sampling
        x_out, _ = denoise_fireflow(self.model, **inp_target, timesteps=timesteps,
                                   guidance=guidance, inverse=False, info=info)

        # Decode
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x_out.device)

        batch_x = unpack(x_out.float(), new_h, new_w)
        x = batch_x[0].unsqueeze(0)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        # Convert to PIL
        x = x.clamp(-1, 1)
        from einops import rearrange
        x = rearrange(x[0], "c h w -> h w c")
        result = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return result


# Convenience function
def edit_image(
    source_image: str | np.ndarray | Image.Image,
    source_prompt: str,
    target_prompt: str,
    **kwargs
) -> Image.Image:
    """
    Quick image editing function.

    Args:
        source_image: Source image
        source_prompt: Source description
        target_prompt: Target description
        **kwargs: Additional arguments for AdaEditPipeline.edit()

    Returns:
        Edited PIL Image
    """
    pipeline = AdaEditPipeline()
    return pipeline.edit(source_image, source_prompt, target_prompt, **kwargs)
