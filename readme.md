<!-- ![]() -->
# StyleFaceUV &mdash; Official PyTorch implementation
This repository is the official implementation of **StyleFaceUV: A 3D Face UV Map Generator for View-Consistent Face Image Synthesis**.

[[Paper (BMVC)]](https://bmvc2022.mpi-inf.mpg.de/0089.pdf)[[Project page]](https://bmvc2022.mpi-inf.mpg.de/89/)

# Environment setting
Tested on Nvidia V100S, Ubuntu 18.04.5 LTS.

```
conda create -n style_face_uv python=3.8
conda activate style_face_uv
conda install -f environment.yml
```

## Testing
Run the following commands to open Gradio web ui for our demo.

```
python webui.py
```

Note that gradio and gltf doesn't support the external lighting condition, we just demonstrate it with default light setting of them.
![](demo/demo.webm)

## Training
### Dataset preparation
Firstly, we need to make the synthetic dataset to train our "Style to Yaw Angle" and "Style to 3Dcoeffs" and "3D Generator" module.

Clone the [3DMM model fitting using Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch) repository and place it elsewhere.

Here, we placed this repository at "this_repo/../" folder so that their fit_single_img.py is in "this_repo/../3DMM-Fitting-Pytorch".

After installing their repository following the instruction written in the README.md in their repo, runs the following command in "this_repo/".

```
python make_dataset.py
```

To accelerate the data fetch process, we further run the following command to aggregate the synthesized dataset into lmdb form.

```
python prepare_data.py data/train_stylegan_images/
```

### Train the "Style to 3Dcoeffs" module
Making sure that you've downloaded the external required data at ??? and placed them into "this_repo/data" directory.

Run the following commands to train our "Style to 3Dcoeffs" module, the checkpoints will be placed in "ckpt/stylecode_to_3dmm_coeff" directory.

```
python trainer_3dcoeff_lightning.py
```

### Extract "Yaw pose" editing direction for pre-trained StyleGAN2 on FHHQ dataset
To extract the yaw angle editing direction, we followed the approach proposed by [InterFaceGAN](https://github.com/genforce/interfacegan).

Simply runs the following command to extract yaw pose editing direction to "data/pose_direction-new.pkl"

```
python trainer_pose_boundary.py
```

### Train the "Style to Yaw Angle" module
To estimate the quantity for "Yaw pose" editing direction to rotate image to its symmetric view, we additionally trained our "Style to Yaw Angle" module.

Run the following commands to train our "Style to Yaw Angle" module, the checkpoints will be placed in "ckpt/stylecode_to_yaw_angle" directory.

```
python trainer_yaw_pose_scalar_lightning.py
```

### Train the "3D Generator" module
Finally, we utilize modified StyleGAN2 to provided diffuse map and displacement map for corresponding style code.

Run the following commands to train our "3D Generator" module, the checkpoints will be placed in "ckpt/style_face_uv" directory.

```
python trainer_3d_lightning.py
```

## Acknowledgement
[StyleGAN2 &mdash; Pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by rosinality

[Pytorch plugins provided by StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

[3DMM model fitting using Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch)

[InterFaceGAN](https://github.com/genforce/interfacegan)

## Citation

```bibtex
@inproceedings{Chung_2022_BMVC,
author    = {Wei-Chieh Chung and Jian-Kai Zhu and I-Chao Shen and Yu-Ting Wu and Yung-Yu Chuang},
title     = {StyleFaceUV: a 3D Face UV Map Generator for View-Consistent Face Image Synthesis},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0089.pdf}
}
```
