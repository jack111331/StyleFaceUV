import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils,transforms
from .network.arch import Generator2D, Generator3D
from tqdm import tqdm
import cv2
from scipy.io import loadmat
import time, copy
import os, math
import os.path as path
import torch.nn.functional as F
from .network.arch import StylecodeTo3DMMCoeffMLP, StylecodeToPoseDirectionScalarMLP
from .network.arch_3DMM import Pure_3DMM_UV
from operator import itemgetter
import random
##############################################################################################
#########               主要的inference code , 用以測試訓練結果                       ########
###########################################################################################
def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    return uv_coords
def multipose(coeff, inputs, pure_3dmm, num, uv_coord, uv_coord_2, uvmap, uv_displace_map):
    coeff[0][224] = torch.tensor(0).cuda()
    coeff[0][226] = torch.tensor(0).cuda()
    for angle in [-60,0,60]:
        coeff[0][225] = torch.tensor(angle*math.pi/180).cuda()
        x = {'coeff': coeff,
             'uvmap': uvmap,
             'uv_displace_map': uv_displace_map,
             'uv_coord': uv_coord,
             'uv_coord_2': uv_coord_2}
        rendered_img, _, _, _, _, _ = pure_3dmm(x)
        rendered_img = rendered_img.detach().cpu().numpy()[0]
        a = np.clip(rendered_img[:, :, :3],0,1)
        out_img = (a*255).astype(np.uint8)
        cv2.imwrite(path.join(args.output_dir, 'test%02d-angle%d.png'%(num, angle)),out_img[:,:,::-1])
if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument("--FACE_MODEL_PATH",type=str,default='./BFM/BFM_model_front.mat')
    parser.add_argument("--load_model",type=str,default='./checkpoints/sep_deeper/param22.pkl', help='StyleCode to 3D Coef.')
    parser.add_argument("--Wplus", type=str, default='./pkls/tensor-new.pkl', help="path of W tensor.")
    parser.add_argument("--coeffs", type=str, default='./pkls/3DMMparam-new.pkl', help="path of 3DMM coeff tensor.")
    parser.add_argument("--multi_pose", action="store_true",default=False)
    parser.add_argument("--output_uv", action="store_true",default=False)
    parser.add_argument("--cover_ori", action="store_true",default=False)
    parser.add_argument("--add_flip", action="store_true",default=False)
    parser.add_argument("--gray", action="store_true",default=False)
    parser.add_argument("--output_dir", type=str, default='./for_ppt', help="output_dir")
    parser.add_argument("--load_model_3d", type=str, default='checkpoints_final/param.pkl', help='Our final checkpoints')
    args = parser.parse_args()
    if not path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ####### Original StyleGAN2
    g_ema_orig= Generator2D(256, 512, 8, channel_multiplier=2).cuda()
    g_ema_orig.load_state_dict(torch.load("./models/StyleGAN2_ffhq_resol=256_550000.pt")["g_ema"],strict=False)
    #################################################
    ###### StyleCode to 3DMM
    model = StylecodeTo3DMMCoeffMLP().cuda()
    model.load_state_dict(torch.load(args.load_model))
    ###################################################
    #####  3DMM with UV Map 
    pure_3dmm = Pure_3DMM_UV(loadmat(args.FACE_MODEL_PATH)).cuda()
    ##################################################
    ####   Our 3D Generator which outputs UV Maps
    g_ema= Generator3D(256, 512, 8, channel_multiplier=2).cuda()
    g_ema.load_state_dict(torch.load(args.load_model_3d))
    ###################################################
    ####  Load Pre-processed stylecode (wplus) and corresponding 3D coef.
    wplus = torch.load(args.Wplus)
    coeffs = torch.load(args.coeffs)
    #pose_direction = torch.load('./pkls/pose_direction-new.pkl').view(14,512).type(torch.FloatTensor).cuda()
    #################################################################
    ####  Some pre-defined Grid-Sampling settings , u dont need to manage it
    uv_coord = torch.unsqueeze(torch.tensor(loadmat('data/BFM_UV_refined.mat')['UV_refine']), 0).type('torch.DoubleTensor').cuda()
    uv_coord = uv_coord.repeat(1, 1, 1) # (b,35709,2)
    uv_coord_2 = torch.tensor(loadmat('data/BFM_UV_refined.mat')['UV_refine'])
    uv_coord_2 = process_uv(uv_coord_2)
    uv_coord_2 = uv_coord_2[:,:2]/255.
    uv_coord_2 = uv_coord_2*2 - 1
    uv_coord_2 = torch.unsqueeze(uv_coord_2, 0).type('torch.DoubleTensor').cuda()
    uv_coord_2 = uv_coord_2.repeat(1,1,1)
    uv_coord_2 = torch.unsqueeze(uv_coord_2, 1)
    ###################################################################
    for idx, num in enumerate(np.arange(5).tolist()):
        # inputs = torch.unsqueeze(wplus[num], 0).cuda() # 從training data裡拿
        inputs = g_ema_orig.get_latent_Wplus(torch.randn(1,512).cuda())         # 隨機生成
        #################################################
        ########     輸出StyleGAN image        ##########
        ################################################
        image, _ = g_ema_orig(inputs, input_is_Wplus=True)
        image = (image+1)/2
        image = image.permute(0,2,3,1)*255
        image = image.detach().cpu().numpy()[0]
        image = image[:,:,::-1]
        cv2.imwrite(path.join(args.output_dir, 'test%02d-ori.png'%(num)),image)
        ######################################################
        #coeff = torch.unsqueeze(coeffs[rand_num], 0).cuda()
        coeff = model(torch.flatten(inputs, start_dim=1)) # 輸出預測的3D Coef.
        uvmap, uv_displace_map = g_ema(inputs,  input_is_Wplus=True, return_uv=True)  #輸出 UV Maps
        uvmap = torch.tanh(uvmap)
        uvmap = (uvmap +1)/2       
        if(args.gray == True):                                #     純輸出3D Shape
            uvmap = (torch.ones(1,3,256,256)*0.5).cuda()      #     texture設成灰色
            coeff[:,227:254] = (torch.ones(1,27)*0.15).cuda() #     亮度固定
        ###########################################
        uvmap = uvmap.permute(0,2,3,1)
        x = {'coeff': coeff,
             'uvmap': uvmap,
             'uv_displace_map': uv_displace_map,
             'uv_coord': uv_coord,
             'uv_coord_2': uv_coord_2}
        rendered_img, _, _, _, _, _ = pure_3dmm(x)  # Render出2D Image
        rendered_img = rendered_img.detach().cpu().numpy()[0]
        a = np.clip(rendered_img[:, :, :3],0,1)
        out_img = (a*255).astype(np.uint8)
        out_img = out_img[:,:,::-1]
        if args.cover_ori:    # 黏上原本StyleGAN的背景 (只限原來的pose)
            out_mask = (rendered_img[:, :, 3]>0).astype(np.uint8)     
            out_img = cv2.bitwise_and(image,image,mask=1-out_mask) + cv2.bitwise_and(out_img,out_img,mask=out_mask)
        cv2.imwrite(path.join(args.output_dir, 'test%02d.png'%(num)),out_img)
        if(args.output_uv):   # 輸出UV Texture map
            out_uvmap = uvmap.detach().cpu().numpy()[0]
            out_uvmap = np.clip(out_uvmap, 0, 1)
            out_uvmap = (out_uvmap*255).astype(np.uint8)
            out_uvmap = out_uvmap[:,:,::-1]
            cv2.imwrite(path.join(args.output_dir,'test%02d-uv.png'%(num)), out_uvmap)
        if(args.multi_pose):  # 輸出不同view
            multipose(coeff, inputs, pure_3dmm, num, uv_coord, uv_coord_2, uvmap, uv_displace_map)
