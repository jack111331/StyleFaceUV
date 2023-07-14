import os

from torch.utils.data import DataLoader
import torchvision
import numpy as np
from PIL import Image

from utils.utility import instantiate_from_config
from dataloader.dataset_lightning import StyleCodeImage3DMMParamsPoseDirDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import yaml

class LogImageCallback(Callback):
    def __init__(self, log_image_period_step=100):
        self.log_image_period_step = log_image_period_step

    def log_images(self, pl_module, batch, batch_idx):
        if hasattr(pl_module, "log_images") and callable(pl_module.log_images):
            pl_module.eval()
            root = os.path.join(pl_module.logger.save_dir, "images")

            imgs, data_imgs, diffuse_map, displacement_map = pl_module.log_images(batch)
            if True:
                grid = torchvision.utils.make_grid(imgs, nrow=4)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).cpu()
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "rendered_{}_gs-b-{:06}.png".format(
                    pl_module.global_step,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
            if True:
                grid = torchvision.utils.make_grid(data_imgs, nrow=4)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).cpu()
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "sampled_{}_gs-b-{:06}.png".format(
                    pl_module.global_step,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
            if True:
                grid = torchvision.utils.make_grid(diffuse_map, nrow=4)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).cpu()
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "diffuse_{}_gs-b-{:06}.png".format(
                    pl_module.global_step,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
            if True:
                grid = torchvision.utils.make_grid(displacement_map, nrow=4)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).cpu()
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "dispalcement_{}_gs-b-{:06}.png".format(
                    pl_module.global_step,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, train_batch, batch_idx, dataloader_idx):
        if batch_idx % self.log_image_period_step == 0:
            self.log_images(pl_module, train_batch, batch_idx)

class SetupCallback(Callback):
    def __init__(self, resume, ckptdir):
        super().__init__()
        self.resume = resume
        self.ckptdir = ckptdir

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

if __name__ == '__main__':
    with open('config/config.yml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_config = config['model']

    model = instantiate_from_config(model_config)
    model.learning_rate = model_config['learning_rate']
    model.n_critic = model_config['n_critic']
    model.n_critic_d = model_config['n_critic_d']
    model.d_reg_every = model_config['d_reg_every']
    model.photo_weight = model_config['photo_weight']

    batch_size = 12
    dataset_dir = "./data/"

    dataset = StyleCodeImage3DMMParamsPoseDirDataset(dataset_dir, clean=True)
    data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)
    # FIXME prepare another validation dataset, sample 1K StyleGAN2 image

    val_data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)

    trainer = pl.Trainer(default_root_dir="ckpt/style_face_uv", gpus=1, max_epochs=25, callbacks=[LogImageCallback(), SetupCallback(True, ckptdir='ckpt/style_face_uv/test_ckpt')])
    trainer.fit(model, data_loaders, val_data_loaders)

