import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils,transforms
from .network.arch import Generator2D, Generator3D
from tqdm import tqdm
import cv2, copy
from scipy.io import loadmat
import time, copy
import os, math
import os.path as path
import torch.nn.functional as F
from .network.arch import StylecodeTo3DMMCoeffMLP, StylecodeToPoseDirectionScalarMLP
from .network.arch_3DMM import Pure_3DMM_UV
from operator import itemgetter
import random
from imageio import mimwrite
#################################################################
#########               輸出動圖用                       ########
###############################################################
def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    return uv_coords
if __name__ == "__main__":
    device = "cuda"
    #torch.cuda.set_device(2)
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument("--FACE_MODEL_PATH",type=str,default='./BFM/BFM_model_front.mat')
    parser.add_argument("--load_model",type=str,default='./checkpoints/sep_deeper/param22.pkl')
    parser.add_argument("--Wplus", type=str, default='./pkls/tensor-new.pkl', help="path of W tensor.")
    parser.add_argument("--coeffs", type=str, default='./pkls/3DMMparam-new.pkl', help="path of 3DMM coeff tensor.")
    parser.add_argument("--output_dir", type=str, default='./for_ppt', help="Num of Examples taken to fit SVM")
    parser.add_argument("--load_model_3d", type=str, default='checkpoints_final/param.pkl')
    args = parser.parse_args()
    if not path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    g_ema_orig= Generator2D(256, 512, 8, channel_multiplier=2).cuda()
    g_ema_orig.load_state_dict(torch.load("./models/StyleGAN2_ffhq_resol=256_550000.pt")["g_ema"],strict=False)
    model = StylecodeTo3DMMCoeffMLP().cuda()
    model.load_state_dict(torch.load(args.load_model))
    pure_3dmm = Pure_3DMM_UV(loadmat(args.FACE_MODEL_PATH)).cuda()
    wplus = torch.load(args.Wplus)
    coeffs = torch.load(args.coeffs)
    #pose_direction = torch.load('./pkls/pose_direction.pkl').view(14,512).type(torch.FloatTensor).cuda()
    g_ema= Generator3D(256, 512, 8, channel_multiplier=2).cuda()
    g_ema.load_state_dict(torch.load(args.load_model_3d))
    uv_coord = torch.unsqueeze(torch.tensor(loadmat('data/BFM_UV_refined.mat')['UV_refine']), 0).type('torch.DoubleTensor').cuda()
    uv_coord = uv_coord.repeat(1, 1, 1) # (b,35709,2)
    uv_coord_2 = torch.tensor(loadmat('data/BFM_UV_refined.mat')['UV_refine'])
    uv_coord_2 = process_uv(uv_coord_2)
    uv_coord_2 = uv_coord_2[:,:2]/255.
    uv_coord_2 = uv_coord_2*2 - 1
    uv_coord_2 = torch.unsqueeze(uv_coord_2, 0).type('torch.DoubleTensor').cuda()
    uv_coord_2 = uv_coord_2.repeat(1,1,1)
    uv_coord_2 = torch.unsqueeze(uv_coord_2, 1)
    for num in np.arange(1).tolist():
        inputs = g_ema_orig.get_latent_Wplus(torch.randn(1,512).cuda())
        coeff = model(torch.flatten(inputs, start_dim=1))
        ######################################################
        image, _ = g_ema_orig(inputs, input_is_Wplus=True)
        image = (image+1)/2
        image = image.permute(0,2,3,1)*255
        image = image.detach().cpu().numpy()[0]
        image = image[:,:,::-1]
        cv2.imwrite(path.join(args.output_dir, 'test%02d-ori.png'%(num)),image)
        ######################################################
        coeff = model(torch.flatten(inputs, start_dim=1))
        uvmap, uv_displace_map = g_ema(inputs,  input_is_Wplus=True, return_uv=True)
        uvmap = torch.tanh(uvmap)
        uvmap = (uvmap +1)/2
        uvmap = uvmap.permute(0,2,3,1)
        angles = np.linspace(-1,1,41).tolist() + np.linspace(1,-1,41).tolist()
        recon_rotate = []
        for ind, angle in enumerate(angles):
            coeff[0][225] = torch.tensor(angle).cuda()
            x = {'coeff': coeff,
                 'uvmap': uvmap,
                 'uv_displace_map': uv_displace_map,
                 'uv_coord': uv_coord,
                 'uv_coord_2': uv_coord_2}
            rendered_img, _, _, _, _, _ = pure_3dmm(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            a = np.clip(rendered_img[:, :, :3],0,1)
            out_img = (a*255).astype(np.uint8)
            #out_img = out_img[:,:,::-1]
            recon_rotate.append(out_img)
        recon_rotate = np.array(recon_rotate)
        #print(recon_rotate.shape)
        recon_rotate = torch.tensor(recon_rotate)
        mimwrite(path.join(args.output_dir, 'test%02d.gif'%(num)), recon_rotate)


