import glob
import json
import os
import skimage
import numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from relighting.image_processor import pil_square_image
from tqdm.auto import tqdm
import random
import itertools
from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self,
                 resolution=(1024, 1024),
                 force_square=True,
                 return_image_path=False,
                 return_dict=False,
        ):
        """
        Resoution is (WIDTH, HEIGHT)
        """
        self.resolution = resolution
        self.force_square = force_square
        self.return_image_path = return_image_path
        self.return_dict = return_dict
        self.scene_data = []
        self.meta_data = []
        self.boundary_info = []
        
    @abstractmethod
    def _load_data_path(self):
        pass

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        image = Image.open(self.scene_data[idx])
        if self.force_square:
            image = pil_square_image(image, self.resolution)
        else:
            image = image.resize(self.resolution)
        
        if self.return_dict:
            d = {
                "image": image,
                "path": self.scene_data[idx]
            }
            if len(self.boundary_info) > 0:
                d["boundary"] = self.boundary_info[idx]
                
            return d
        elif self.return_image_path:
            return image, self.scene_data[idx]
        else:
            return image

class GeneralLoader(Dataset):
    def __init__(self,
                 root=None,
                 num_samples=None,
                 res_threshold=((1024, 1024)),
                 apply_threshold=False,
                 random_shuffle=False,
                 process_id = 0,
                 process_total = 1,
                 limit_input = 0,
                 **kwargs,
        ):
        super().__init__(**kwargs)
        self.root = root
        self.res_threshold = res_threshold
        self.apply_threshold = apply_threshold
        self.has_meta = False
        
        if self.root is not None:
            if not os.path.exists(self.root):
                raise Exception(f"Dataset {self.root} does not exist.") 
            
            paths = natsorted(
                list(glob.glob(os.path.join(self.root, "*.png"))) + \
                list(glob.glob(os.path.join(self.root, "*.jpg")))
            )
            self.scene_data = self._load_data_path(paths, num_samples=num_samples)
            
            if random_shuffle:
                SEED = 0
                random.Random(SEED).shuffle(self.scene_data)
                random.Random(SEED).shuffle(self.boundary_info)
            
            if limit_input > 0:
                self.scene_data = self.scene_data[:limit_input]
                self.boundary_info = self.boundary_info[:limit_input]
                
            # please keep this one the last, so, we will filter out scene_data and boundary info
            if process_total > 1:
                self.scene_data = self.scene_data[process_id::process_total]
                self.boundary_info = self.boundary_info[process_id::process_total]
                print(f"Process {process_id} has {len(self.scene_data)} samples")

    def _load_data_path(self, paths, num_samples=None):
        if os.path.exists(os.path.splitext(paths[0])[0] + ".json") or os.path.exists(os.path.splitext(paths[-1])[0] + ".json"):
            self.has_meta = True
        
        if self.has_meta:
            # read metadata
            TARGET_KEY = "chrome_mask256"
            for path in paths:
                with open(os.path.splitext(path)[0] + ".json") as f:
                    meta = json.load(f)
                    self.meta_data.append(meta)
                    boundary =  {
                        "x": meta[TARGET_KEY]["x"],
                        "y": meta[TARGET_KEY]["y"],
                        "size": meta[TARGET_KEY]["w"],
                    }
                    self.boundary_info.append(boundary)
                
        
        scene_data = paths
        if self.apply_threshold:
            scene_data = []
            for path in tqdm(paths):
                img = Image.open(path)
                if (img.size[0] >= self.res_threshold[0]) and (img.size[1] >= self.res_threshold[1]):
                    scene_data.append(path)
        
        if num_samples is not None:
            max_idx = min(num_samples, len(scene_data))
            scene_data = scene_data[:max_idx]
        
        return scene_data
    
    @classmethod
    def from_image_paths(cls, paths, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        dataset.scene_data = dataset._load_data_path(paths)
        return dataset

class ALPLoader(Dataset):
    def __init__(self,
                 root=None,
                 num_samples=None,
                 res_threshold=((1024, 1024)),
                 apply_threshold=False,
                 **kwargs,
        ):
        super().__init__(**kwargs)
        self.root = root
        self.res_threshold = res_threshold
        self.apply_threshold = apply_threshold
        self.has_meta = False
        
        if self.root is not None:
            if not os.path.exists(self.root):
                raise Exception(f"Dataset {self.root} does not exist.") 
            
            dirs = natsorted(list(glob.glob(os.path.join(self.root, "*"))))
            self.scene_data = self._load_data_path(dirs)

    def _load_data_path(self, dirs):
        self.scene_names = [Path(dir).name for dir in dirs]

        scene_data = []
        for dir in dirs:
            pseudo_probe_dirs = natsorted(list(glob.glob(os.path.join(dir, "*"))))
            pseudo_probe_dirs = [dir for dir in pseudo_probe_dirs if "gt" not in dir]
            data = [os.path.join(dir, "images", "0.png") for dir in pseudo_probe_dirs]
            scene_data.append(data)

        scene_data = list(itertools.chain(*scene_data))
        return scene_data

class MultiIlluminationLoader(Dataset):
    def __init__(self,
                root, 
                mask_probe=True, 
                mask_boundingbox=False,
                **kwargs,
        ):
        """
        @params resolution (tuple): (width, height) - resolution of the image
        @params force_square: will add black border to make the image square while keeping the aspect ratio
        @params mask_probe: mask the probe with the mask in the dataset
        
        """
        super().__init__(**kwargs)
        self.root = root
        self.mask_probe = mask_probe
        self.mask_boundingbox = mask_boundingbox

        if self.root is not None:
            dirs = natsorted(list(glob.glob(os.path.join(self.root, "*"))))
            self.scene_data = self._load_data_path(dirs)

    def _load_data_path(self, dirs):
        self.scene_names = [Path(dir).name for dir in dirs]

        data = {}
        for dir in dirs:
            chrome_probes = natsorted(list(glob.glob(os.path.join(dir, "probes", "*chrome*.jpg"))))
            gray_probes = natsorted(list(glob.glob(os.path.join(dir, "probes", "*gray*.jpg"))))
            scenes = natsorted(list(glob.glob(os.path.join(dir, "dir_*.jpg"))))

            with open(os.path.join(dir, "meta.json")) as f:
                meta_data = json.load(f)
            
            bbox_chrome = meta_data["chrome"]["bounding_box"]
            bbox_gray = meta_data["gray"]["bounding_box"]

            mask_chrome = os.path.join(dir, "mask_chrome.png")
            mask_gray = os.path.join(dir, "mask_gray.png")

            scene_name = Path(dir).name
            data[scene_name] = {
                "scenes": scenes,
                "chrome_probes": chrome_probes,
                "gray_probes": gray_probes,
                "bbox_chrome": bbox_chrome,
                "bbox_gray": bbox_gray,
                "mask_chrome": mask_chrome,
                "mask_gray": mask_gray,
            }
        return data

    def _mask_probe(self, image, mask):
        """
        mask probe with a png file in dataset
        """
        image_anticheat = skimage.img_as_float(np.array(image))
        mask_np = skimage.img_as_float(np.array(mask))[..., None]
        image_anticheat = ((1.0 - mask_np) * image_anticheat) + (0.5 * mask_np)
        image_anticheat = Image.fromarray(skimage.img_as_ubyte(image_anticheat))
        return image_anticheat
    
    def _mask_boundingbox(self, image, bbox): 
        """
        mask image with the bounding box for anti-cheat
        """
        bbox = {k:int(np.round(v/4.0)) for k,v in bbox.items()}
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        image_anticheat = skimage.img_as_float(np.array(image))
        image_anticheat[y:y+h, x:x+w] = 0.5
        image_anticheat = Image.fromarray(skimage.img_as_ubyte(image_anticheat))
        return image_anticheat
        
    def __getitem__(self, scene_name):
        data = self.scene_data[scene_name]
        
        mask_chrome = Image.open(data["mask_chrome"])
        mask_gray = Image.open(data["mask_gray"])
        images = []
        for path in data["scenes"]:
            image = Image.open(path)
            if self.mask_probe:
                image = self._mask_probe(image, mask_chrome)
                image = self._mask_probe(image, mask_gray)
            if self.mask_boundingbox:
                image = self._mask_boundingbox(image, data["bbox_chrome"])
                image = self._mask_boundingbox(image, data["bbox_gray"])
                
            if self.force_square:
                image = pil_square_image(image, self.resolution)
            else:
                image = image.resize(self.resolution)
            images.append(image)

        chrome_probes = [Image.open(path) for path in data["chrome_probes"]]
        gray_probes = [Image.open(path) for path in data["gray_probes"]]
        bbox_chrome = data["bbox_chrome"]
        bbox_gray = data["bbox_gray"]
        
        return images, chrome_probes, gray_probes, bbox_chrome, bbox_gray

    
    def calculate_ball_info(self, scene_name):
        # TODO: remove hard-coded parameters
        ball_data = []
        for mtype in ['bbox_chrome', 'bbox_gray']:
            info = self.scene_data[scene_name][mtype]

            # x-y is top-left corner of the bounding box
            # meta file is for 4000x6000 image but dataset is 1000x1500
            x = info['x'] / 4
            y = info['y'] / 4
            w = info['w'] / 4
            h = info['h'] / 4

           
            # we scale data to 512x512 image 
            if self.force_square:
                h_ratio = (512.0 * 2.0 / 3.0) / 1000.0    #384 because we have black border on the top
                w_ratio = 512.0 / 1500.0
            else:
                h_ratio = self.resolution[0] / 1000.0
                w_ratio = self.resolution[1] / 1500.0
                
            x = x * w_ratio
            y = y * h_ratio
            w = w * w_ratio
            h = h * h_ratio

            if self.force_square:
                # y need to shift due to top black border
                top_border_height = 512.0 * (1/6)
                y = y + top_border_height


            # Sphere is not circle due to the camera perspective, Need future fix for this
            # For now, we use the minimum of width and height
            w = int(np.round(w))
            h = int(np.round(h))
            if w > h:
                r = h
                x = x + (w - h) / 2.0
            else:
                r = w 
                y = y + (h - w) / 2.0                
            
            x = int(np.round(x))
            y = int(np.round(y))
            
            ball_data.append((x, y, r))
        
        return ball_data
    
    def calculate_bbox_info(self, scene_name):
        # TODO: remove hard-coded parameters
        bbox_data = []
        for mtype in ['bbox_chrome', 'bbox_gray']:
            info = self.scene_data[scene_name][mtype]

            # x-y is top-left corner of the bounding box
            # meta file is for 4000x6000 image but dataset is 1000x1500
            x = info['x'] / 4
            y = info['y'] / 4
            w = info['w'] / 4
            h = info['h'] / 4

           
            # we scale data to 512x512 image 
            if self.force_square:
                h_ratio = (512.0 * 2.0 / 3.0) / 1000.0    #384 because we have black border on the top
                w_ratio = 512.0 / 1500.0
            else:
                h_ratio = self.resolution[0] / 1000.0
                w_ratio = self.resolution[1] / 1500.0
                
            x = x * w_ratio
            y = y * h_ratio
            w = w * w_ratio
            h = h * h_ratio

            if self.force_square:
                # y need to shift due to top black border
                top_border_height = 512.0 * (1/6)
                y = y + top_border_height


            w = int(np.round(w))
            h = int(np.round(h))             
            x = int(np.round(x))
            y = int(np.round(y))
            
            bbox_data.append((x, y, w, h))
        
        return bbox_data

    """
    DO NOT remove this!
    This is for evaluating results from Multi-Illumination generated from the old version
    """
    def calculate_ball_info_legacy(self, scene_name):
        # TODO: remove hard-coded parameters
        ball_data = []
        for mtype in ['bbox_chrome', 'bbox_gray']:
            info = self.scene_data[scene_name][mtype]

            # x-y is top-left corner of the bounding box
            # meta file is for 4000x6000 image but dataset is 1000x1500
            x = info['x'] / 4
            y = info['y'] / 4
            w = info['w'] / 4
            h = info['h'] / 4

            # we scale data to 512x512 image 
            h_ratio = 384.0 / 1000.0    #384 because we have black border on the top
            w_ratio = 512.0 / 1500.0
            x = x * w_ratio
            y = y * h_ratio
            w = w * w_ratio
            h = h * h_ratio

            # y need to shift due to top black border
            top_border_height = 512.0 * (1/8)

            y = y + top_border_height

            # Sphere is not circle due to the camera perspective, Need future fix for this
            # For now, we use the minimum of width and height
            r = np.max(np.array([w, h]))  

            x = int(np.round(x))
            y = int(np.round(y))
            r = int(np.round(r))

            ball_data.append((y, x, r))

        return ball_data