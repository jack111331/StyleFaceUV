<!-- ![]() -->
# StyleFaceUV &mdash; Official PyTorch implementation
This repository is the official implementation of **StyleFaceUV: A 3D Face UV Map Generator for View-Consistent Face Image Synthesis**.

# Environment setting
Tested on Nvidia V100S
```
conda create -n style_face_uv python=3.8
conda activate style_face_uv
conda install -f environment.yml
```

## Testing
## Training
### Dataset preparation
Firstly, we need to make the synthetic dataset to train our "Style to Yaw Angle" and "Style to 3Dcoeffs" and "3D Generator" module.

Clone the [3DMM model fitting using Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch) repository and place it elsewhere.

Here, we placed this repository at "this_repo/../" folder so that their fit_single_img.py is in "this_repo/../3DMM-Fitting-Pytorch".

After installing their repository following the instruction written in the README.md in their repo, runs the following command in "this_repo/".

```
python make_dataset.py
```

### Train the "Style to 3Dcoeffs" module
Making sure that you've download the external required data at ??? and placed them into "this_repo/data" directory.



### Extract "Yaw pose" editing direction
To extract the yaw angle editing direction, we followed the approach proposed by [InterFaceGAN](https://github.com/genforce/interfacegan).

Simply runs the following command to extract yaw pose editing direction to "./data/pose_direction-new.pkl"

```
python trainer_pose_boundary.py
```

### Train the "Style to Yaw Angle" module

### Train the "3D Generator" module


## Acknowledgement
[StyleGAN2 &mdash; Pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by rosinality

[Pytorch plugins provided by StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

[3DMM model fitting using Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch)

[InterFaceGAN](https://github.com/genforce/interfacegan)
## Contact