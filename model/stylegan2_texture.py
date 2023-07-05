import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from model.stylegan2 import *

import pytorch_lightning as PL

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
            if(ind == length -1):
                uv_skip = self.uv_conv(out, latent[:,i+2], skip)
                uv_skip = self.tanh(uv_skip)
            skip = skip_ori
            i += 2

        image = skip
        uv_image = uv_skip
        if return_latents:
            return image, latent
        elif return_uv:
            return image, uv_image
        else:
            return image, None

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

# FIXME refactor
class StylecodeTo3DMMCoeffMLP(nn.Module):
    def __init__(self, MM_param_num=257):
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

    def forward(self, stylecode):
        all_wo_tex = self.Net1(stylecode)
        tex = self.Net2(stylecode)
        return torch.cat((all_wo_tex[:, :144], tex, all_wo_tex[:, 144:]), dim=1)


# 從 Split_coeff()開始，mostly get from Deep3DFaceReconstruction/reconstruct_mesh.py

class StylecodeToPoseDirectionScalarMLP(nn.Module):
    def __init__(self, param_num=1):
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
