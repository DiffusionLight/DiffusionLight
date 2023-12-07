import torch
import numpy as np
from PIL import Image
from scipy.special import sph_harm

def crop_ball(image, mask_ball, x, y, size, apply_mask=True, bg_color = (0, 0, 0)):
    if isinstance(image, Image.Image):
        result = np.array(image)
    else:
        result = image.copy()
    
    result = result[y:y+size, x:x+size]
    if apply_mask:
        result[~mask_ball] = bg_color
    return result

def get_ideal_normal_ball(size, flip_x=True):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen    
    (we flip X to match sobel operator)
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x

    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def get_predicted_normal_ball(size, precomputed_path=None):
    if precomputed_path is not None:
        normal_map = Image.open(precomputed_path).resize((size, size))
        normal_map = np.array(normal_map).astype(np.uint8)
        _, mask = get_ideal_normal_ball(size)
    else:
        raise NotImplementedError

    normal_map = (normal_map - 127.5) / 127.5 # normalize for compatibility with inpainting pipeline
    return normal_map, mask