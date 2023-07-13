import torchvision
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import importlib

import os

def output_grid_img_from_tensor(img_tensor, img_path):
    # img_tensor size (B, C, H, W)
    grid_img_tensor = torchvision.utils.make_grid(img_tensor, nrow=4, normalize=True, range=(-1, 1), scale_each=True)
    # Change from C x H x W to H x W x C
    img = grid_img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img * 255)

def output_img_from_tensor(img_tensor, img_path, index=0):
    # img_tensor size (B, C, H, W)
    img_tensor = (img_tensor + 1) / 2  # from (-1,1) to (0,1)

    # Change from C x H x W to H x W x C
    img_tensor = torch.round(img_tensor.permute(0, 2, 3, 1) * 255)
    img = cv2.cvtColor(img_tensor.detach().cpu().numpy()[index], cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)
    return img

def output_3dmm_img_from_tensor(img_tensor, img_path):
    rendered_img = img_tensor.detach().cpu().numpy()[0]
    out_img = rendered_img[:, :, :3].astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, out_img)

def output_3dmm_masked_img_from_tensor(img_tensor, img_path, other_img):
    rendered_img = img_tensor.detach().cpu().numpy()[0]
    out_img = rendered_img[:, :, :3].astype(np.uint8)
    out_mask = (rendered_img[:, :, 3] > 0).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_img = cv2.bitwise_and(other_img, other_img, mask=1 - out_mask) + cv2.bitwise_and(out_img, out_img, mask=out_mask)
    cv2.imwrite(img_path, out_img)

def train_valid_split(dataset, proportion, batch_size):
    datasets = {'train': dataset[:int(len(dataset) * proportion)], 'val': dataset[int(len(dataset) * proportion):]}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, drop_last=True) for x in
                   ['train', 'val']}
    return datasets, dataloaders

def ensure_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Code borrowed from https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)