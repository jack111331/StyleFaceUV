import math
import random
import os

import torch
from torch import nn
from torch.autograd import Variable
import einops

from utils.losses import *
import lpips

from model.stylegan2 import *
from utils.utility import instantiate_from_config

import pytorch_lightning as pl

class Generator3D(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.uv_conv = ToRGB(128, style_dim)
        self.tanh = nn.Tanh()

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)
    
    def get_latent_Wplus(self,input):
        styles = self.style(input)
        styles = [styles]
        latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        return latent
    
    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        input_is_Wplus=False,
        return_uv = False
    ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]
        if(input_is_Wplus==False):
            if not input_is_latent:
                styles = [self.style(s) for s in styles]
            if truncation < 1:
                style_t = []

                for style in styles:
                    style_t.append(
                        truncation_latent + truncation * (style - truncation_latent)
                    )

                styles = style_t
            if len(styles) < 2: #1
                inject_index = self.n_latent
                #print(styles[0].ndim)
                if styles[0].ndim < 3: #2 if W
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1) #(b, 14, 512)
                else: #3 if W+
                    latent = styles[0]

            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)
        else:
            latent = styles

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        length = len(list(zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs)))
        for ind, (conv1, conv2, noise1, noise2, to_rgb) in enumerate(zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        )):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip_ori = to_rgb(out, latent[:, i + 2], skip)
            if ind == (length - 1):
                uv_skip = self.uv_conv(out, latent[:,i+2], skip)
            skip = skip_ori
            i += 2

        diffuse_map = self.tanh(skip)
        displacement_map = self.tanh(uv_skip)
        if return_latents:
            return diffuse_map, latent
        elif return_uv:
            return diffuse_map, displacement_map
        else:
            return diffuse_map, None

class Discriminator3D(nn.Module):
    def __init__(self, size=256, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

# This module should setup:
'''
    this_model.learning_rate: float | Learning rate for both generator and discriminator  
    this_model.n_critic: int | Optimize generator every G step
    this_model.n_critic_d: int | Optimize discriminator every D step
    this_model.d_reg_every: int | Apply regularization every R step
    this_model.photo_weight: float | Photo loss weighting
'''
class StyleFaceUV(pl.LightningModule):
    def __init__(self, 
                 size, 
                 latent_dim, 
                 n_mlp, 
                 gan_ckpt_path, 
                 style_to_pose_scalar_config,
                 style_to_3dmm_coeff_config,
                 face_renderer_config,
                 pose_direction_path,
                 uvmask_path,
                 ckpt_path=None):
        
        super().__init__()
        self.generator_3d = Generator3D(size, latent_dim, n_mlp)
        self.discriminator_3d = Discriminator3D()
        self.generator_2d = Generator2D(size, latent_dim, n_mlp)
        gan_ckpt = torch.load(gan_ckpt_path)
        self.generator_3d.load_state_dict(gan_ckpt['g_ema'], strict=False)
        self.discriminator_3d.load_state_dict(gan_ckpt['d'], strict=False)
        self.generator_2d.load_state_dict(gan_ckpt['g_ema'], strict=False)
        self.generator_2d.eval()

        self.stylecode_to_scalar_model = instantiate_from_config(style_to_pose_scalar_config).eval()
        self.stylecode_to_3dmm_coeff_model = instantiate_from_config(style_to_3dmm_coeff_config).eval()
        self.pure_3dmm_model = instantiate_from_config(face_renderer_config).eval()

        pose_direction = nn.Parameter(torch.load(pose_direction_path).view(14, 512).type(torch.FloatTensor), requires_grad=False)
        self.register_parameter('pose_direction', pose_direction)

        uvmask = cv2.imread(uvmask_path, 0) > 0
        uvmask = nn.Parameter(torch.tensor(uvmask)[None, ...], requires_grad=False)
        self.register_parameter('uvmask', uvmask)

        self.PerceptLoss = lpips.LPIPS(net='vgg')
        
        self.loss_dict = {}
        self.save_hyperparameters()

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])

    @property
    def automatic_optimization(self):
        return False
    
    def configure_optimizers(self):
        # The self.learning_rate is configured after instantiating this model
        generator_params = list(self.generator_3d.parameters())
        optimizer_g = torch.optim.Adam(generator_params, lr=self.learning_rate)

        discriminator_params = list(self.discriminator_3d.parameters())
        optimizer_d = torch.optim.Adam(discriminator_params, lr=self.learning_rate)
        return [optimizer_g, optimizer_d], []
    
    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def save_as_gltf(self, style_code):
        from utils.gltf_utils import output_mesh_to_gltf
        coeff_3dmm = self.stylecode_to_3dmm_coeff_model(torch.flatten(style_code, start_dim=1))
        _, _, diffuse_map, displacement_map = self(style_code, coeff_3dmm)
        vertices, indices, uv_coord = self.pure_3dmm_model.output_mesh(coeff_3dmm, displacement_map)
        os.makedirs("output", exist_ok=True)
        output_mesh_to_gltf(vertices, indices, uv_coord, diffuse_map, "output/generated_face.gltf")

    def forward(self, style_code, coeff_3dmm, diffuse_map=None, displacement_map=None, output_texture_maps=True):
        if diffuse_map is None or displacement_map is None:
            generated_texture_maps = self.generator_3d(style_code, truncation=1, truncation_latent=None, input_is_Wplus=True,
                                        return_uv=True)
        if diffuse_map is None:
            diffuse_map = generated_texture_maps[0]
            diffuse_map = (diffuse_map + 1) / 2

        if displacement_map is None:
            displacement_map = generated_texture_maps[1]

        rendered_img, _, _, _, _, _ = self.pure_3dmm_model(coeff_3dmm, diffuse_map, displacement_map)
        rendered_mask = rendered_img[:, :, :, 3:4].detach()
        rendered_mask = (rendered_mask > 0)
        rendered_mask = einops.rearrange(rendered_mask, 'b h w c -> b c h w')
        recon_image = rendered_img[:, :, :, :3]
        recon_image = torch.clamp(recon_image, 0, 1)
        recon_image = einops.rearrange(recon_image, 'b h w c -> b c h w')
        if output_texture_maps == True:
            return recon_image, rendered_mask, diffuse_map, displacement_map
        else:
            return recon_image, rendered_mask

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        optimizer_g, optimizer_d = self.optimizers()

        style_code, sampled_image, coeff_3dmm = train_batch
        sampled_image = sampled_image.float()

        sym_sampled_image, sym_coeff_3dmm, sym_gradleftmask = self.synthesize_sym_view_img(style_code, coeff_3dmm)
        if batch_idx % self.n_critic_d == 0:
            # Discriminator step
            self.requires_grad(self.generator_3d, False)
            self.requires_grad(self.discriminator_3d, True)
            optimizer_d.zero_grad()

            recon_image, rendered_mask, diffuse_map, displacement_map = self(style_code, coeff_3dmm, output_texture_maps=True)
            masked_sample_image = sampled_image * rendered_mask.repeat((1, 3, 1, 1))

            real_pred = self.discriminator_3d(masked_sample_image * 2 - 1)
            fake_pred = self.discriminator_3d(recon_image * 2 - 1)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            sym_recon_image, rendered_sym_mask = self(None, sym_coeff_3dmm.detach(), diffuse_map=diffuse_map, displacement_map=displacement_map, output_texture_maps=False)
            masked_sym_sampled_img = sym_sampled_image * rendered_sym_mask
            sym_real_pred = self.discriminator_3d(masked_sym_sampled_img * 2 - 1)
            sym_fake_pred = self.discriminator_3d(sym_recon_image * 2 - 1)
            sym_d_loss = d_logistic_loss(sym_real_pred, sym_fake_pred)

            total_d_loss = 0.75 * sym_d_loss + d_loss
            self.manual_backward(total_d_loss)
            self.loss_dict.update({"train/d_loss": total_d_loss})
            optimizer_d.step()

            if batch_idx % (self.n_critic_d * self.d_reg_every) == 0:
                optimizer_d.zero_grad()

                masked_sample_image = sampled_image * rendered_mask.repeat((1, 3, 1, 1))
                masked_sample_image = (masked_sample_image * 2) - 1
                masked_sample_image.requires_grad = True
                real_pred = self.discriminator_3d(masked_sample_image)
                r1_loss = d_r1_loss(real_pred, masked_sample_image)

                sym_masked_sample_image = sym_sampled_image * rendered_sym_mask
                sym_masked_sample_image = (sym_masked_sample_image * 2) - 1
                sym_masked_sample_image.requires_grad = True
                sym_real_pred = self.discriminator_3d(sym_masked_sample_image)
                sym_r1_loss = d_r1_loss(sym_real_pred, sym_masked_sample_image)

                self.discriminator_3d.zero_grad()
                total_d_reg_loss = (10 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]) + 0.75 * (
                        10 / 2 * sym_r1_loss * self.d_reg_every + 0 * sym_real_pred[0])
                self.manual_backward(total_d_reg_loss)
                self.loss_dict.update({"train/d_reg_loss": total_d_reg_loss})
                optimizer_d.step()

        if batch_idx % self.n_critic == 0:
            # Generator step
            self.requires_grad(self.generator_3d, True)
            self.requires_grad(self.discriminator_3d, False)
            optimizer_g.zero_grad()
            recon_image, rendered_mask, diffuse_map, displacement_map = self(style_code, coeff_3dmm)

            fake_pred = self.discriminator_3d(recon_image * 2 - 1)
            g_loss = g_nonsaturating_loss(fake_pred)

            stylerig_photo_loss_v = photo_loss(einops.rearrange(recon_image, 'b c h w -> b h w c'), (einops.rearrange(sampled_image, 'b c h w -> b h w c')).detach(), torch.squeeze(rendered_mask, dim=1))

            image_percept = sampled_image * rendered_mask.repeat(1, 3, 1, 1)
            perceptual_loss_v = torch.mean(self.PerceptLoss(image_percept * 2 - 1, recon_image * 2 - 1))

            sym_gradleftmask = torch.unsqueeze(sym_gradleftmask, 1).repeat(1, 3, 1, 1)
            weighted_mask, _, _, _, _, _ = self.pure_3dmm_model(sym_coeff_3dmm, sym_gradleftmask, displacement_map, need_illu=False)
            weighted_mask = torch.clamp(weighted_mask, 0, 1)
            weighted_mask = weighted_mask[..., 0].detach()

            sym_recon_image, rendered_sym_mask = self(None, sym_coeff_3dmm.detach(), diffuse_map=diffuse_map, displacement_map=displacement_map, output_texture_maps=False)
            sym_fake_pred = self.discriminator_3d(sym_recon_image * 2 - 1)
            sym_g_loss = g_nonsaturating_loss(sym_fake_pred)

            sym_stylerig_photo_loss_v = photo_loss(einops.rearrange(sym_recon_image, 'b c h w -> b h w c'), (einops.rearrange(sym_sampled_image, 'b c h w -> b h w c')).detach(), weighted_mask)
            sym_sampled_image_percept = sym_sampled_image * rendered_sym_mask.repeat(1, 3, 1, 1)
            sym_perceptual_loss_v = torch.mean(self.PerceptLoss(sym_sampled_image_percept * 2 - 1, sym_recon_image * 2 - 1))

            loss = 0.75 * sym_g_loss + g_loss + self.photo_weight * (
                    stylerig_photo_loss_v + 0.75 * sym_stylerig_photo_loss_v + 0.2 * perceptual_loss_v + 0.2 * 0.75 * sym_perceptual_loss_v)
            self.manual_backward(loss)
            self.loss_dict.update({"train/g_loss": loss})
            optimizer_g.step()

        self.log_dict(self.loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def synthesize_sym_view_img(self, style_code, coeff_3dmm):
        batch_size = style_code.shape[0]
        scalar = self.stylecode_to_scalar_model(torch.flatten(style_code, start_dim=1)).view(-1, 1, 1)

        symmetric_style_code = style_code + scalar * self.pose_direction.detach()
        sym_sampled_image_2d, _ = self.generator_2d(symmetric_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
        sym_sampled_image_2d = torch.clamp(sym_sampled_image_2d, -1, 1)
        sym_sampled_image_2d = ((sym_sampled_image_2d + 1) / 2).detach()
        sym_coeff_3dmm = self.stylecode_to_3dmm_coeff_model(torch.flatten(symmetric_style_code, start_dim=1))

        sym_gradmask = self.get_gradmask(sym_coeff_3dmm.detach()).detach()
        gradmask = self.get_gradmask(coeff_3dmm.detach()).detach()  # (b, 256, 256) 0~1
        sym_gradleftmask = (sym_gradmask - (sym_gradmask > 0) * (gradmask > 0) * 1.)
        # simplify to this.
        # sym_gradleftmask = (self.uvmask > 0) * (~(sym_gradleftmask > 0)) * 0.5 + sym_gradleftmask
        uvmask = self.uvmask.repeat(batch_size, 1, 1)
        sym_gradleftmask = (uvmask > 0) * (~((uvmask > 0) * (sym_gradleftmask > 0))) * 0.5 + sym_gradleftmask
        return sym_sampled_image_2d, sym_coeff_3dmm, sym_gradleftmask

    def get_gradmask(self, coeff_3dmm):
        gradient_map = Variable(torch.ones(coeff_3dmm.shape[0], 3, 256, 256) * 100, requires_grad=False).to(
            self.device)
        grad_displacement_map = Variable(torch.ones(coeff_3dmm.shape[0], 3, 256, 256) * 0, requires_grad=True).to(
            self.device)
        grad_rendered_img, _, _, _, _, _ = self.pure_3dmm_model(coeff_3dmm.detach(), gradient_map, grad_displacement_map)
        grad_rendered_img = torch.clamp(grad_rendered_img, 0, 255)
        grad_displacement_map.retain_grad()
        (torch.sum(grad_rendered_img[:, :, :, :3] * 255)).backward()
        gradmask = torch.sum(torch.abs(grad_displacement_map.grad), dim=1)  # (b, 256, 256)
        gradmask = (gradmask != 0) * 1.
        return gradmask
    
    @torch.no_grad()
    def log_images(self, batch):
        style_code, image, coeff_3dmm = batch
        image = image.float()
        device = self.device
        style_code, image, coeff_3dmm = style_code.to(device), image.to(device), coeff_3dmm.to(device)

        recon_image, _, diffuse_map, displacement_map = self(style_code, coeff_3dmm, output_texture_maps=True)
        return recon_image, image, diffuse_map, displacement_map

    def validation_step(self, batch, batch_idx):
        # First we sample from original stylegan2 and measure reconstruction loss
        # Masked image's SSIM, PSNR, L2, LPIPS
        style_code, sampled_image, coeff_3dmm = batch
        sampled_image = sampled_image.float()

        val_loss_dict = {}

        recon_image, rendered_mask = self(style_code, coeff_3dmm, output_texture_maps=False)
        val_loss_dict["val/l2_loss"] = photo_loss(einops.rearrange(recon_image, 'b c h w -> b h w c'), (einops.rearrange(sampled_image, 'b c h w -> b h w c')).detach(), torch.squeeze(rendered_mask, dim=1))

        image_percept = sampled_image * rendered_mask.repeat(1, 3, 1, 1)
        val_loss_dict["val/percept_loss"] = torch.mean(self.PerceptLoss(image_percept * 2 - 1, recon_image * 2 - 1))

        self.log_dict(val_loss_dict, logger=True, on_epoch=True)

        # TODO In future plan we measure kid, fid to evaluate the model performance

class StylecodeTo3DMMCoeffMLP(nn.Module):
    def __init__(self, ckpt_path=None, MM_param_num=257):
        super(StylecodeTo3DMMCoeffMLP, self).__init__()
        self.Net1 = torch.nn.Sequential(
            nn.Linear(14 * 512, 9 * 512),
            nn.Tanh(),
            nn.Linear(9 * 512, 6 * 512),
            nn.Tanh(),
            nn.Linear(6 * 512, 3 * 512),
            nn.Tanh(),
            nn.Linear(3 * 512, 512),
            nn.Tanh(),
            nn.Linear(512, MM_param_num - 80)
        )
        self.Net2 = torch.nn.Sequential(
            nn.Linear(14 * 512, 9 * 512),
            nn.ELU(),
            nn.Linear(9 * 512, 6 * 512),
            nn.ELU(),
            nn.Linear(6 * 512, 3 * 512),
            nn.ELU(),
            nn.Linear(3 * 512, 512),
            nn.ELU(),
            nn.Linear(512, 80)
        )
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            # TODO This should be deprecated, and setup another mechanism to record the optimizer
            if 'Stylecode to 3DMM Coeff' in ckpt:
                self.load_state_dict(ckpt['Stylecode to 3DMM Coeff'])
            else:
                new_state_dict = {}
                for k in ckpt['state_dict'].keys():
                    if k.startswith('stylecode_to_3dmm_coeff_model'):
                        new_state_dict[k[len('stylecode_to_3dmm_coeff_model')+1:]] = ckpt['state_dict'][k]

                self.load_state_dict(new_state_dict)

    def forward(self, stylecode):
        all_wo_tex = self.Net1(stylecode)
        tex = self.Net2(stylecode)
        return torch.cat((all_wo_tex[:, :144], tex, all_wo_tex[:, 144:]), dim=1)


# 從 Split_coeff()開始，mostly get from Deep3DFaceReconstruction/reconstruct_mesh.py

class StylecodeToPoseDirectionScalarMLP(nn.Module):
    def __init__(self, ckpt_path=None, param_num=1):
        super(StylecodeToPoseDirectionScalarMLP, self).__init__()
        self.Net = torch.nn.Sequential(
            nn.Linear(14 * 512, 9 * 512),
            nn.Tanh(),
            nn.Linear(9 * 512, 6 * 512),
            nn.Tanh(),
            nn.Linear(6 * 512, 3 * 512),
            nn.Tanh(),
            nn.Linear(3 * 512, 512),
            nn.Tanh(),
            nn.Linear(512, param_num)
        )
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            # TODO This should be deprecated, and setup another mechanism to record the optimizer
            if 'Stylecode to Pose Scalar' in ckpt:
                self.load_state_dict(ckpt['Stylecode to Pose Scalar'])
            else:
                new_state_dict = {}
                for k in ckpt['state_dict'].keys():
                    if k.startswith('stylecode_to_scalar_model'):
                        new_state_dict[k[len('stylecode_to_scalar_model')+1:]] = ckpt['state_dict'][k]

                self.load_state_dict(new_state_dict)

    def forward(self, stylecode):
        return self.Net(stylecode)


class StylecodeToMultiPoseDirectionScalarMLP(nn.Module):
    def __init__(self, param_num=1):
        super(StylecodeToMultiPoseDirectionScalarMLP, self).__init__()
        self.Net = torch.nn.Sequential(
            nn.Linear(14 * 512 + 1000, 9 * 512),
            nn.Tanh(),
            nn.Linear(9 * 512, 6 * 512),
            nn.Tanh(),
            nn.Linear(6 * 512, 3 * 512),
            nn.Tanh(),
            nn.Linear(3 * 512, 512),
            nn.Tanh(),
            nn.Linear(512, param_num)
        )
        self.Net2 = torch.nn.Sequential(
            nn.Linear(1, 1000),
            nn.Tanh()
        )

    def forward(self, stylecode, pose):
        y = self.Net2(pose)
        x = torch.cat([stylecode, y], 1)
        return self.Net(x)
