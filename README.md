# DiffusionLight: Light Probes for Free by Painting a Chrome Ball

### [Project Page](https://diffusionlight.github.io/) | [Paper](https://arxiv.org/abs/2312.09168) | [Colab](https://colab.research.google.com/drive/15pC4qb9mEtRYsW3utXkk-jnaeVxUy-0S?usp=sharing&sandboxMode=true) | [HuggingFace](https://huggingface.co/DiffusionLight/DiffusionLight)

[![Open DiffusionLight in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15pC4qb9mEtRYsW3utXkk-jnaeVxUy-0S?usp=sharing&sandboxMode=true)

![](https://diffusionlight.github.io/assets/images/thumbnail.jpg)

 We present a simple yet effective technique to estimate lighting in a single input image. Current techniques rely heavily on HDR panorama datasets to train neural networks to regress an input with limited field-of-view to a full environment map. However, these approaches often struggle with real-world, uncontrolled settings due to the limited diversity and size of their datasets. To address this problem, we leverage diffusion models trained on billions of standard images to render a chrome ball into the input image. Despite its simplicity, this task remains challenging: the diffusion models often insert incorrect or inconsistent objects and cannot readily generate images in HDR format. Our research uncovers a surprising relationship between the appearance of chrome balls and the initial diffusion noise map, which we utilize to consistently generate high-quality chrome balls. We further fine-tune an LDR diffusion model (Stable Diffusion XL) with LoRA, enabling it to perform exposure bracketing for HDR light estimation. Our method produces convincing light estimates across diverse settings and demonstrates superior generalization to in-the-wild scenarios. 

 ## Table of contents
-----
  * [TL;DR](#Getting-started)
  * [Installation](#Installation)
  * [Prediction](#Prediction)
  * [Evaluation](#Evaluation)
  * [Citation](#Citation)
------

## Getting started

```shell
conda env create -f environment.yml
conda activate diffusionlight
pip install -r requirements.txt
python inpaint.py --dataset example --output_dir output
python ball2envmap.py --ball_dir output/square --envmap_dir output/envmap
python exposure2hdr.py --input_dir output/envmap --output_dir output/hdr
```

## Installation

To setup the Python environment, you need to run the following commands in both Conda and pip:

```shell
conda env create -f environment.yml
conda activate diffusionlight
pip install -r requirements.txt
```

Note that Conda is optional. However, if you choose not to use Conda, you must manually install CUDA-toolkit and OpenEXR.

## Prediction

### 0. Preparing the image

Please resize the input image to 1024x1024. If the image is not square, we recommend padding it with a black border.

### 1. Inpainting the chrome ball

First, we predict the chrome ball in different exposure values (EV) using the following command:

```shell
python inpaint.py --dataset <input_directory> --output_dir <output_directory>
```

This command outputs three subdirectories:  `control`, `raw`, and  `square`

The contents of each directory are:

- `control`: Conditioned depth map
- `raw`: Inpainted image with a chrome ball in the center
- `square`: Square-cropped chrome ball (used for the next step)


### 2. Projecting a ball into an environment map 

Next, we project the chrome ball from the previous step to the LDR environment map using the following command:

```shell
python ball2envmap.py --ball_dir <output_directory>/square --envmap_dir <output_directory>/envmap
```

### 3. Compose HDR image

Finally, we compose an HDR image from multiple LDR environment maps using our custom exposure bracketing:

```shell
python exposure2hdr.py --input_dir <output_directory>/envmap --output_dir <output_directory>/hdr
```

The predicted light estimation will be located at `<output_directory>/hdr` and can be used for downstream tasks such as object insertion. We will also use it to compare with other methods.

## Evaluation 
We use the evaluation code from [StyleLight](https://style-light.github.io/) and [Editable Indoor LightEstimation](https://lvsn.github.io/EditableIndoorLight/). You can use their code to measure our score.

Additionally, we provide a *slightly* modified version of the evaluation code at [DiffusionLight-evaluation](https://github.com/DiffusionLight/DiffusionLight-evaluation) including the test input.

## Citation

```
@inproceedings{Phongthawee2023DiffusionLight,
    author = {Phongthawee, Pakkapon and Chinchuthakun, Worameth and Sinsunthithet, Nontaphat and Raj, Amit and Jampani, Varun and Khungurn, Pramook and Suwajanakorn, Supasorn},
    title = {DiffusionLight: Light Probes for Free by Painting a Chrome Ball},
    booktitle = {ArXiv},
    year = {2023},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)
