import torch
import numpy as np
import itertools
from diffusers.utils.torch_utils import randn_tensor

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def expand_noise(noise, shape, seed, device, dtype):
    new_generator = torch.Generator().manual_seed(seed)
    corner_shape = (shape[0], shape[1], shape[2] // 2, shape[3] // 2)
    vert_border_shape = (shape[0], shape[1], shape[2], shape[3] // 2)
    hori_border_shape = (shape[0], shape[1], shape[2] // 2, shape[3])

    corners = [randn_tensor(corner_shape, generator=new_generator, device=device, dtype=dtype) for _ in range(4)]
    vert_borders = [randn_tensor(vert_border_shape, generator=new_generator, device=device, dtype=dtype) for _ in range(2)]
    hori_borders = [randn_tensor(hori_border_shape, generator=new_generator, device=device, dtype=dtype) for _ in range(2)]

    # combine
    big_shape = (shape[0], shape[1], shape[2] * 2, shape[3] * 2)
    noise_template = randn_tensor(big_shape, generator=new_generator, device=device, dtype=dtype)

    ticks = [(0, 0.25), (0.25, 0.75), (0.75, 1.0)]
    grid = list(itertools.product(ticks, ticks))
    noise_list = [
        corners[0], hori_borders[0], corners[1],
        vert_borders[0], noise, vert_borders[1],
        corners[2], hori_borders[1], corners[3],
    ]
    for current_noise, ((x1, x2), (y1, y2)) in zip(noise_list, grid):
        top_left = (int(x1 * big_shape[2]), int(y1 * big_shape[3]))
        bottom_right = (int(x2 * big_shape[2]), int(y2 * big_shape[3]))
        noise_template[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = current_noise

    return noise_template

def custom_prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        use_noise_moving=True,
        return_noise=False,
        return_image_latents=False,
        newx=0,
        newy=0,
        newr=256,
        current_seed=None,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)

        if latents is None and use_noise_moving:
            # random big noise map
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            noise = expand_noise(noise, shape, seed=current_seed, device=device, dtype=dtype)
 
            # ensure noise is the same regardless of inpainting location (top-left corner notation)
            newys = [newy] if not isinstance(newy, list) else newy
            newxs = [newx] if not isinstance(newx, list) else newx
            big_noise = noise.clone()
            prev_noise = None
            for newy, newx in zip(newys, newxs):
                # find patch location within big noise map
                sy = big_noise.shape[2] // 4 + ((512 - 128) - newy) // self.vae_scale_factor
                sx = big_noise.shape[3] // 4 + ((512 - 128) - newx) // self.vae_scale_factor

                if prev_noise is not None:
                    new_noise = big_noise[:, :, sy:sy+shape[2], sx:sx+shape[3]]

                    ball_mask = torch.zeros(shape, device=device, dtype=bool)
                    top_left = (newy // self.vae_scale_factor, newx // self.vae_scale_factor)
                    bottom_right = (top_left[0] + newr // self.vae_scale_factor, top_left[1] + newr // self.vae_scale_factor) # fixed ball size r = 256
                    ball_mask[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = True

                    noise = prev_noise.clone()
                    noise[ball_mask] = new_noise[ball_mask]
                else:
                    noise = big_noise[:, :, sy:sy+shape[2], sx:sx+shape[3]]

                prev_noise = noise.clone()

            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs
    
def custom_prepare_mask_latents(
    self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
):
    # resize the mask to latents shape as we concatenate the mask to the latents
    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
    # and half precision
    mask = torch.nn.functional.interpolate(
        mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
        mode="bilinear", align_corners=False #PURE: We add this to avoid sharp border of the ball
    )
    mask = mask.to(device=device, dtype=dtype)

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    if mask.shape[0] < batch_size:
        if not batch_size % mask.shape[0] == 0:
            raise ValueError(
                "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                " of masks that you pass is divisible by the total requested batch size."
            )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

    masked_image_latents = None
    if masked_image is not None:
        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

    return mask, masked_image_latents