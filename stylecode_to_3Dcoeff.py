import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, copy, cv2
import os.path as path
from dataloader.dataset import StyleCode3DMMParamsDataset
from .network.arch import StylecodeTo3DMMCoeffMLP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.losses import l1loss
from scipy.io import loadmat
from .network.arch import Generator2D
from .network.arch_3DMM import Pure_3DMM
#####################################################
###########         StyleCode to 3D Coeffs.    ######
#####################################################
def inference(args,model, g_ema, pure_3dmm):
    model.eval()
    g_ema.eval()
    pure_3dmm.eval()
    for i in range(10):
        wplus = g_ema.get_latent_Wplus(torch.randn(1,512).cuda())
        coeff = model(torch.flatten(wplus, start_dim=1))
        image, _ = g_ema(wplus, truncation=1, truncation_latent=None, input_is_Wplus=True)
        image = (image +1)/2 # from (-1,1) to (0,1)
        image = torch.round(image.permute(0,2,3,1)*255)
        image = image.detach().cpu().numpy()[0][:,:,::-1]
        cv2.imwrite(os.path.join(args.output_dir,'stylegan_image%02d.png'%(i)),image)
        x = {'coeff': coeff}
        rendered_img, _, _, _, _,_ = pure_3dmm(x)
        rendered_img = rendered_img.detach().cpu().numpy()[0]
        out_img = rendered_img[:, :, :3].astype(np.uint8)
        out_mask = (rendered_img[:, :, 3]>0).astype(np.uint8)
        out_img = out_img[:,:,::-1]
        cv2.imwrite(os.path.join(args.output_dir,'3dmm_image%02d.png'%(i)),out_img)

    return
def train(args,dataloaders, model, criterion, optimizer, scheduler,g_ema,pure_3dmm):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    check_data = torch.unsqueeze(torch.load('./pkls/tensor-new.pkl')[0], 0).cuda()
    g_ema.eval()
    pure_3dmm.eval()
    for epoch in range(1,args.num_epochs+1):
        print('Epoch {}/{}'.format(epoch, args.num_epochs ))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for idx,data in tqdm(enumerate(dataloaders[phase]),total=len(dataloaders[phase])):
                style_code = data[0].cuda()
                coeff = data[1].cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(torch.flatten(style_code, start_dim=1))
                    #loss = criterion(output, coeff)
                    optimizer.zero_grad()
                    loss = l1loss(output,coeff)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders[phase])
            print("Phase: %s, Epoch: %02d, Loss: %f"%(phase, epoch, epoch_loss))
            image, _ = g_ema(check_data, truncation=1, truncation_latent=None, input_is_Wplus=True)
            image = (image +1)/2 # from (-1,1) to (0,1)
            image = torch.round(image.permute(0,2,3,1)*255)
            image = image.detach().cpu().numpy()[0][:,:,::-1]
            cv2.imwrite(os.path.join(args.output_dir,'stylegan_image.png'),image)
            output = model(torch.flatten(check_data, start_dim=1))
            x = {'coeff': output}
            rendered_img, _, _, _, _, _ = pure_3dmm(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            out_img = rendered_img[:, :, :3].astype(np.uint8)
            out_mask = (rendered_img[:, :, 3]>0).astype(np.uint8)
            out_img = out_img[:,:,::-1]
            out_img = cv2.bitwise_and(image,image,mask=1-out_mask) + cv2.bitwise_and(out_img,out_img,mask=out_mask)
            cv2.imwrite(os.path.join(args.output_dir,'3dmm_image%02d.png'%(epoch)),out_img)
            if(phase=='val' and epoch_loss < best_loss):
                print("Got New Best !")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path.join(args.output_dir,"param%02d.pkl"%(epoch)))
        if scheduler is not None:
            scheduler.step()
    return
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_rate",type=float,default=1e-5)
    parser.add_argument("--num_epochs",type=int,default=25)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--output_dir",type=str,default='./checkpoints')
    parser.add_argument("--ckpt",type=str,default="./models/StyleGAN2_ffhq_resol=256_550000.pt",help="path to the model checkpoint")
    parser.add_argument("--channel_multiplier",type=int,default=2,help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--latent",type=int,default=512)
    parser.add_argument("--n_mlp",type=int,default=8)
    parser.add_argument("--FACE_MODEL_PATH",type=str,default='./BFM/BFM_model_front.mat')
    parser.add_argument("--mode", type=str, default='train')
    return parser.parse_args()
def main():
    args = get_args()
    args.output_dir = 'handover'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = StylecodeTo3DMMCoeffMLP().cuda()
    g_ema = Generator2D(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).cuda()
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    try:
        facemodel = loadmat(args.FACE_MODEL_PATH)
    except Exception as e:
        print('failed to load %s' %(args.FACE_MODEL_PATH))
    # FIXME assert facemodel
    pure_3dmm = Pure_3DMM(facemodel).cuda()
    if args.mode == 'train':
        dataset = StyleCode3DMMParamsDataset('./pkls')
        datasets = {'train':dataset[:int(len(dataset)*0.9)],'val':dataset[int(len(dataset)*0.9):]}
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True) for x in ['train','val']}
        criterion = nn.L1Loss().cuda()#nn.MSELoss().cuda()
        optimizer = optim.Adam(model.parameters(),lr=args.l_rate)
        scheduler = MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
        train(args,dataloaders, model, criterion, optimizer, scheduler,g_ema,pure_3dmm)
    elif args.mode=='test':
        model.load_state_dict(torch.load('./checkpoints/sep_deeper/param22.pkl'))
        inference(args, model, g_ema, pure_3dmm)
if __name__ =='__main__':
    main()