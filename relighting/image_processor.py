import torch
import numpy as np
from PIL import Image, ImageChops
import skimage
try:
    import cv2
except:
    pass

def fill_image(image, mask_ball, x, y, size, color=(255,255,255)):
    if isinstance(image, Image.Image):
        result = np.array(image)
    else:
        result = image.copy()
    
    result[y:y+size, x:x+size][mask_ball] = color
    
    if isinstance(image, Image.Image):
        result = Image.fromarray(result)

    return result

def pil_square_image(image, desired_size = (512,512), interpolation=Image.LANCZOS):
    """
    Make top-bottom border
    """
    # Don't resize if already desired size (Avoid aliasing problem)
    if image.size == desired_size:
        return image
    
    # Calculate the scale factor
    scale_factor = min(desired_size[0] / image.width, desired_size[1] / image.height)

    # Resize the image
    resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), interpolation)

    # Create a new blank image with the desired size and black border
    new_image = Image.new("RGB", desired_size, color=(0, 0, 0))

    # Paste the resized image onto the new image, centered
    new_image.paste(resized_image, ((desired_size[0] - resized_image.width) // 2, (desired_size[1] - resized_image.height) // 2))
    
    return new_image

# https://stackoverflow.com/questions/19271692/removing-borders-from-an-image-in-python
def remove_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)

# Taken from https://huggingface.co/lllyasviel/sd-controlnet-normal
def estimate_scene_normal(image, depth_estimator):
    # can be improve speed do not going back and float between numpy and torch
    normal_image = depth_estimator(image)['predicted_depth'][0]

    normal_image = normal_image.numpy()

    # upsizing image depth to match input
    hw = np.array(image).shape[:2]
    normal_image = skimage.transform.resize(normal_image, hw, preserve_range=True)

    image_depth = normal_image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)
    
    bg_threhold = 0.4

    x = cv2.Sobel(normal_image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(normal_image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    normal_image = np.stack([x, y, z], axis=2)
    normal_image /= np.sum(normal_image ** 2.0, axis=2, keepdims=True) ** 0.5

    # rescale back to image size
    return normal_image

def estimate_scene_depth(image, depth_estimator):
    #image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    #with torch.no_grad(), torch.autocast("cuda"):
    #    depth_map = depth_estimator(image).predicted_depth

    depth_map = depth_estimator(image)['predicted_depth']
    W, H = image.size
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def fill_depth_circular(depth_image, x, y, r):
    depth_image = np.array(depth_image)

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            xy = (i - x - r//2)**2 + (j - y - r//2)**2
            # if xy <= rr**2:
            # depth_image[j, i, :] = 255
            # depth_image[j, i, :] = int(minv + (maxv - minv) * z)
            if xy <= (r // 2)**2:
                depth_image[j, i, :] = 255
    
    depth_image = Image.fromarray(depth_image)
    return depth_image
    

def merge_normal_map(normal_map, normal_ball,  mask_ball, x, y):
    """
    Merge a ball to normal map using mask
    @params
        normal_amp (np.array) - normal map of the scene [height, width, 3]
        normal_ball (np.array) - normal map of the ball [ball_height, ball_width, 3]
        mask_ball (np.array) - mask of the ball [ball_height, ball_width]
        x (int) - x position of the ball (top-left)
        y (int) - y position of the ball (top-left)
    @return
        normal_mapthe merge normal map [height, width, 3]        
    """
    result = normal_map.copy()

    mask_ball = mask_ball[..., None]
    ball = (normal_ball * mask_ball) # alpha blending the ball
    unball = (normal_map[y:y+normal_ball.shape[0], x:x+normal_ball.shape[1]] * (1 - mask_ball)) # alpha blending the normal map
    result[y:y+normal_ball.shape[0], x:x+normal_ball.shape[1]] =  ball+unball # add them together
    return result