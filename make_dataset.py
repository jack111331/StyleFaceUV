from model.stylegan2 import *
import torch
from PIL import Image
import numpy as np
import os

# *** Configure yourself *** 
fitting_3dmm_repo_dir = '../3DMM-Fitting-Pytorch'
gan_ckpt_path = 'ckpt/StyleGAN2_ffhq_resol=256_550000.pt'
data_path = 'data/train_stylegan_images'
fitted_result_path = 'data/fitted_result'

output_3dmm_path = 'data/3DMMparam-new.pkl'
output_style_code_path = 'data/tensor-new.pkl'
sample_images_size = 35820

val_data_path = 'data/val_stylegan_images'
val_sample_images_size = 10000


if __name__ == '__main__':
    generator_2d = Generator2D(size=256, style_dim=512, n_mlp=8)
    gan_ckpt = torch.load(gan_ckpt_path)
    generator_2d.load_state_dict(gan_ckpt['g_ema'], strict=False)
    generator_2d = generator_2d.to('cuda')

    # Synthesize training samples
    style_codes = torch.zeros((sample_images_size, 14, 512))

    for i in range(sample_images_size):
        # Generate StyleGAN2 sampled image
        noise = torch.randn(1, 512, device='cuda')
        sampled_style_code = generator_2d.get_latent_Wplus(noise)
        style_codes[i] = sampled_style_code.cpu()
        sampled_img, _ = generator_2d(sampled_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
        sampled_img = sampled_img.permute(0, 2, 3, 1)
        sampled_img = ((sampled_img.detach().cpu().numpy()[0] + 1) / 2 * 255).astype(np.uint8)
        filename = f"{i:05}.png"
        path = os.path.join(data_path, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(sampled_img).save(path)

        # Fit 3DMM parameter on the sampled image
        import subprocess
        prompt = ['python', 'fit_single_img.py',  '--img_path', os.path.relpath(path, start=fitting_3dmm_repo_dir), '--res_folder', os.path.relpath(fitted_result_path, start=fitting_3dmm_repo_dir)]
        proc = subprocess.Popen(prompt, cwd=fitting_3dmm_repo_dir)
        proc.wait()
    
    coeff_3dmms = torch.zeros((sample_images_size, 257))
    for i in range(sample_images_size):
        fitted_coeff_3dmm = np.load(os.path.join(fitted_result_path, f"{i:05}_coeffs.npy"))
        coeff_3dmms[i] = torch.Tensor(fitted_coeff_3dmm)

    torch.save(style_codes, output_style_code_path)
    torch.save(coeff_3dmms, output_3dmm_path)

    # Synthesize validation samples
    for i in range(val_sample_images_size):
        # Generate StyleGAN2 sampled image
        noise = torch.randn(1, 512, device='cuda')
        sampled_style_code = generator_2d.get_latent_Wplus(noise)
        style_codes[i] = sampled_style_code.cpu()
        sampled_img, _ = generator_2d(sampled_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
        sampled_img = sampled_img.permute(0, 2, 3, 1)
        sampled_img = ((sampled_img.detach().cpu().numpy()[0] + 1) / 2 * 255).astype(np.uint8)
        filename = f"{i:05}.png"
        path = os.path.join(val_data_path, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(sampled_img).save(path)
