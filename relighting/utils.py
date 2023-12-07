import argparse
import os
from pathlib import Path
from PIL import Image
import hashlib

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def save_result(
    image, image_path,
    mask=None, mask_path=None,
    normal=None, normal_path=None,
):
    assert isinstance(image, Image.Image)
    os.makedirs(Path(image_path).parent, exist_ok=True)
    image.save(image_path)

    if (mask is not None) and (mask_path is not None):
        assert isinstance(mask, Image.Image)
        os.makedirs(Path(mask_path).parent, exist_ok=True)
        mask.save(mask_path)

    if (normal is not None) and (normal_path is not None):
        assert isinstance(normal, Image.Image)
        os.makedirs(Path(normal_path).parent, exist_ok=True)
        normal.save(normal_path)
        
def name2hash(name: str):
    """
    @see https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    """
    hash_number = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    return hash_number