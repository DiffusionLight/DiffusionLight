# DiffusionLight: Light Probes for Free by Painting a Chrome Ball

### [Project Page](https://diffusionlight.github.io/) | [Paper](https://arxiv.org/pdf/)

 We present a simple yet effective technique to estimate lighting in a single input image. Current techniques rely heavily on HDR panorama datasets to train neural networks to regress an input with limited field-of-view to a full environment map. However, these approaches often struggle with real-world, uncontrolled settings due to the limited diversity and size of their datasets. To address this problem, we leverage diffusion models trained on billions of standard images to render a chrome ball into the input image. Despite its simplicity, this task remains challenging: the diffusion models often insert incorrect or inconsistent objects and cannot readily generate images in HDR format. Our research uncovers a surprising relationship between the appearance of chrome balls and the initial diffusion noise map, which we utilize to consistently generate high-quality chrome balls. We further fine-tune an LDR difusion model (Stable Diffusion XL) with LoRA, enabling it to perform exposure bracketing for HDR light estimation. Our method produces convincing light estimates across diverse settings and demonstrates superior generalization to in-the-wild scenarios. 

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
python inpaint.py --dataset example/images --output_dir output
python ball2envmap.py --ball_dir output/square --envmap_dir output/envmap
python exposure2hdr.py --input_dir output/envmap --output_dir output/hdr
```

## Installation

To setup python environment need to run both conda and pip with following command

```shell
conda env create -f environment.yml
conda activate diffusionlight
pip install -r requirements.txt
```

Note that Conda is optional but you have to manually install CUDA-toolkit and OpenEXR.

## Predict light estimation
Our code is not end-to-end. So, you have to run 3 commands

### 1. Predict the chrome ball
FIrst we predict the chromeball in the different EV (Exposure compensation Value) by using the comamnd 

```shell
python inpaint.py --dataset <input_directory> --output_dir <output_directory>
```

### 2. Convert from the chrome ball into environment map 
```shell
python ball2envmap.py --ball_dir <output_directory>/square --envmap_dir <output_directory>/envmap
```

### 3. Combine differnet exposure to the HDR environment map 
```shell
python ball2envmap.py --ball_dir <output_directory>/envmap --envmap <output_directory>/hdr
```

```shell
python inpaint.py --dataset example/images --output_dir output
```


The predicted light estimation will locate at `<output_directory>/hdr`

## Evaluation 
We use the envaluation code from StyleLight and EditableIndoor. You can use their code. but we will provide the *slightly* modifly version at [diffusionlight-eval](https://github.com)

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