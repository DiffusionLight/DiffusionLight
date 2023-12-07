import argparse
from diffusers import DDIMScheduler, DDPMScheduler, UniPCMultistepScheduler

def get_control_signal_type(controlnet):
    if "normal" in controlnet:
        return "normal"
    elif "depth" in controlnet:
        return "depth"
    else:
        raise NotImplementedError

SD_MODELS = {
    "sd15_old": "runwayml/stable-diffusion-inpainting",
    "sd15_new": "runwayml/stable-diffusion-inpainting",
    "sd21": "stabilityai/stable-diffusion-2-inpainting",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_fast": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd15_depth": "runwayml/stable-diffusion-inpainting",
}

VAE_MODELS = {
    "sdxl": "madebyollin/sdxl-vae-fp16-fix",
    "sdxl_fast": "madebyollin/sdxl-vae-fp16-fix",
}

CONTROLNET_MODELS = {
    "sd15_old": "fusing/stable-diffusion-v1-5-controlnet-normal",
    "sd15_new": "lllyasviel/control_v11p_sd15_normalbae",
    "sd21": "thibaud/controlnet-sd21-normalbae-diffusers",
    "sdxl": "diffusers/controlnet-depth-sdxl-1.0",
    "sdxl_fast": "diffusers/controlnet-depth-sdxl-1.0-small",
    "sd15_depth": "lllyasviel/control_v11f1p_sd15_depth",
}

SAMPLERS = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "unipc": UniPCMultistepScheduler,
}

DEPTH_ESTIMATOR = "Intel/dpt-hybrid-midas"