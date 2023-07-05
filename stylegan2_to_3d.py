import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os, copy, cv2
import os.path as path
from dataloader.dataset import StyleCodeImage3DMMParamsPoseDirDataset
from .network.arch import StylecodeTo3DMMCoeffMLP, StylecodeToPoseDirectionScalarMLP
from torch.utils.data import DataLoader
from utils.losses import g_nonsaturating_loss, d_logistic_loss, d_r1_loss, photo_loss
from scipy.io import loadmat
from .network.arch import Generator3D, Discriminator3D
from model import Generator2D as Generator_orig
from .network.arch_3DMM import Pure_3DMM_UV
from torch.autograd import Variable
from torchvision.utils import make_grid
import lpips
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords
def get_gradmask(sym_coeff, uv_coord, uv_coord_2, args, pure_3dmm):
    grad_uvmap = Variable(torch.ones(args.batch_size, 3, 256, 256)*100, requires_grad=False).cuda()
    grad_uvmap = grad_uvmap.permute(0,2,3,1)
    grad_uv_displace_map = Variable(torch.ones(args.batch_size, 3, 256, 256)*0, requires_grad=True).cuda()
    x = {'coeff': sym_coeff.detach(),
         'uvmap': grad_uvmap,
         'uv_displace_map': grad_uv_displace_map,
         'uv_coord': uv_coord,
         'uv_coord_2': uv_coord_2}
    grad_rendered_img, _, _, _, _, _ = pure_3dmm(x)
    grad_uv_displace_map.retain_grad()
    (torch.sum(grad_rendered_img[:, :, :, :3]*255)).backward()
    gradmask = torch.sum(torch.abs(grad_uv_displace_map.grad), dim=1) #(b, 256, 256)
    gradmask = (gradmask!=0)*1.
    # gradmask = torch.unsqueeze(gradmask, 1)
    # gradmask = make_grid(gradmask, nrow=4, normalize=False,scale_each=True)
    # gradmask = gradmask.permute(1,2,0).detach().cpu().numpy()
    # cv2.imwrite('./testing/image%04d.png'%(idx),gradmask*255)
    return gradmask
def train(args, dataloaders, g_ema, pure_3dmm, discriminator, optimizer, optimizer_d, latent_to_posedirection_scalar, g_ema_orig, model):
    best_model_wts = copy.deepcopy(g_ema.state_dict())
    check_data = torch.load('./pkls/tensor-new.pkl')[0:16].cuda()
    check_coeff = torch.load('./pkls/3DMMparam-new.pkl')[0:16].cuda()
    pure_3dmm.eval()
    g_ema.train()
    g_ema_orig.eval()
    model.eval()
    latent_to_posedirection_scalar.eval()
    discriminator.train()
    uv_coord = torch.unsqueeze(torch.tensor(loadmat('data/BFM_UV_refined.mat')['UV_refine']), 0).type('torch.DoubleTensor').cuda()
    uv_coord = uv_coord.repeat(args.batch_size, 1, 1)
    uv_coord_2 = torch.tensor(loadmat('data/BFM_UV_refined.mat')['UV_refine'])
    uv_coord_2 = process_uv(uv_coord_2)
    uv_coord_2 = uv_coord_2[:,:2]/255.
    uv_coord_2 = uv_coord_2*2 - 1
    uv_coord_2 = torch.unsqueeze(torch.tensor(uv_coord_2), 0).type('torch.DoubleTensor').cuda()
    uv_coord_2 = uv_coord_2.repeat(args.batch_size,1,1)
    uv_coord_2 = torch.unsqueeze(uv_coord_2, 1)
    uvmask = cv2.imread('./train_stylegan_images/mask.png',0)
    uvmask = uvmask>0
    uvmask = torch.tensor(uvmask).cuda()
    uvmask = torch.unsqueeze(uvmask,0).repeat(args.batch_size, 1, 1)
    uvmask = uvmask.detach()
    d_cnt = 0
    g_cnt = 0
    PerceptLoss = lpips.LPIPS(net='vgg').cuda()
    for epoch in range(1,args.num_epochs+1):
        print('Epoch {}/{}'.format(epoch, args.num_epochs ))
        print('-' * 10)
        running_loss = 0.0
        running_d_loss = 0.0
        running_g_loss = 0.0
        for idx,data in tqdm(enumerate(dataloaders),total=len(dataloaders)):
            style_code = data[0].cuda()
            image = data[1].cuda()
            image = image.float()
            coeff = data[2].cuda()
            pose_direction = data[3].cuda()
            pose_direction = pose_direction.detach()
            ##################################################
            ##### Random Sample Angles with different view
            #####################################################
            # random_angle = random.uniform(0.2618, 0.34906) #15~20度
            # p_or_n = ((((coeff[:,225]>=0)*2)-1)*-1).view(-1,1).detach().cpu()
            # p30 = (torch.ones(args.batch_size, 1)*random_angle*p_or_n).cuda()
            # scalar = latent_to_posedirection_scalar(torch.flatten(style_code, start_dim=1), p30).view(-1,1,1)
            #################################################
            ########  Get angles with symmetric view
            ##################################################
            scalar = latent_to_posedirection_scalar(torch.flatten(style_code, start_dim=1)).view(-1,1,1)
            #####################################################
            symmetric_style_code = style_code + scalar * pose_direction
            sym_image, _ =  g_ema_orig(symmetric_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
            sym_image = torch.clamp(sym_image,-1,1)
            sym_image = ((sym_image +1)/2).detach()
            # grid_img = make_grid(image.detach(), nrow=4, normalize=False,scale_each=True)
            # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
            # grid_img = grid_img[:,:,::-1]
            # cv2.imwrite('./testing/image%04d.png'%(idx),grid_img*255)
            # grid_img = make_grid(sym_image, nrow=4, normalize=False,scale_each=True)
            # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
            # grid_img = grid_img[:,:,::-1]
            # cv2.imwrite('./testing/sym_image%04d.png'%(idx),grid_img*255)
            sym_image_255 = sym_image.permute(0,2,3,1)*255
            sym_coeff = model(torch.flatten(symmetric_style_code, start_dim=1))
            ####################################
            sym_gradmask = get_gradmask(sym_coeff.detach(), uv_coord, uv_coord_2, args, pure_3dmm).detach()
            gradmask = get_gradmask(coeff.detach(), uv_coord, uv_coord_2, args, pure_3dmm).detach() #(b, 256, 256) 0~1
            sym_gradleftmask = (sym_gradmask - (sym_gradmask>0)*(gradmask>0)*1.)
            sym_gradleftmask = (uvmask>0)*(~((uvmask>0)*(sym_gradleftmask>0)))*0.5 + sym_gradleftmask
            # grid_img = make_grid(torch.unsqueeze(sym_gradleftmask,1), nrow=4, normalize=False,scale_each=True)
            # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
            # cv2.imwrite('./testing/mask%04d.png'%(idx),grid_img*255)
            if idx % args.n_critic_d == 0:
                d_cnt += 1
                requires_grad(g_ema, False)
                requires_grad(discriminator, True)
                optimizer_d.zero_grad()
                #uvmap = g_ema(style_code)
                uvmap, uv_displace_map =  g_ema(style_code, truncation=1, truncation_latent=None, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap +1)/2
                uvmap = uvmap.permute(0,2,3,1)
                x = {'coeff': coeff,
                     'uvmap': uvmap,
                     'uv_displace_map': uv_displace_map,
                     'uv_coord': uv_coord,
                     'uv_coord_2': uv_coord_2}
                rendered_img, _, _, _, _, _ = pure_3dmm(x)
                mask = rendered_img[:, :, :, 3].detach()
                mask = (mask > 0)
                mask = mask.unsqueeze(3)
                mask = mask.repeat((1,1,1,3))
                mask = mask.permute(0,3,1,2)
                recon_image = rendered_img[:,:,:,:3].permute(0,3,1,2)
                c = image*mask
                c = (c*2) -1
                real_pred = discriminator(c)
                fake_pred = discriminator(recon_image*2-1)#.detach())
                d_loss = d_logistic_loss(real_pred, fake_pred)
                ###################################
                x = {'coeff': sym_coeff.detach(),
                     'uvmap': uvmap,
                     'uv_displace_map': uv_displace_map,
                     'uv_coord': uv_coord,
                     'uv_coord_2': uv_coord_2}
                sym_rendered_img, _, _, _, _, _ = pure_3dmm(x)
                sym_recon_image = sym_rendered_img[:,:,:,:3].permute(0,3,1,2)
                sym_mask = sym_rendered_img[:, :, :, 3].detach()
                sym_mask = (sym_mask > 0)
                sym_mask = sym_mask.unsqueeze(3)
                sym_mask = sym_mask.repeat((1,1,1,3))
                sym_mask = sym_mask.permute(0,3,1,2)
                sym_tmp_img = sym_image*sym_mask
                # grid_img = make_grid(sym_tmp_img, nrow=4, normalize=False,scale_each=True)
                # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
                # grid_img = grid_img[:,:,::-1]
                # cv2.imwrite('./testing/sym_image-mask%04d.png'%(idx),grid_img*255)
                sym_real_pred = discriminator(sym_tmp_img*2-1)
                sym_fake_pred = discriminator(sym_recon_image*2-1)
                sym_d_loss = d_logistic_loss(sym_real_pred, sym_fake_pred)
                ##################################
                discriminator.zero_grad()
                (0.75*sym_d_loss  + d_loss).backward()
                optimizer_d.step()
                running_d_loss += (d_loss.item() + sym_d_loss.item())
                ####
                if d_cnt % args.d_reg_every == 0:
                    #print("YO")
                    optimizer_d.zero_grad()
                    c2 = image*mask
                    image_masked = (c2*2)-1
                    image_masked.requires_grad=True
                    real_pred = discriminator(image_masked)
                    r1_loss = d_r1_loss(real_pred, image_masked)
                    #######3
                    c3 = sym_image*sym_mask
                    sym_image_masked = (c3*2)-1
                    sym_image_masked.requires_grad=True
                    sym_real_pred = discriminator(sym_image_masked)
                    sym_r1_loss = d_r1_loss(sym_real_pred, sym_image_masked)
                    #######
                    discriminator.zero_grad()
                    ((10/2*r1_loss*args.d_reg_every+0*real_pred[0]) + 0.75*(10/2*sym_r1_loss*args.d_reg_every+0*sym_real_pred[0])).backward()
                    optimizer_d.step()
            if idx % args.n_critic == 0:
                g_cnt += 1
                requires_grad(g_ema, True)
                requires_grad(discriminator, False)
                optimizer.zero_grad()
                uvmap, uv_displace_map =  g_ema(style_code, truncation=1, truncation_latent=None, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap +1)/2
                uvmap = uvmap.permute(0,2,3,1)
                x = {'coeff': coeff,
                     'uvmap': uvmap,
                     'uv_displace_map': uv_displace_map,
                     'uv_coord': uv_coord,
                     'uv_coord_2': uv_coord_2}
                rendered_img, _, _, _, _, _ = pure_3dmm(x)
                mask = rendered_img[:, :, :, 3].detach()
                recon_image = rendered_img[:,:,:,:3].permute(0,3,1,2)
                fake_pred = discriminator(recon_image*2-1)
                g_loss = g_nonsaturating_loss(fake_pred)
                image_tmp = (image*255).permute(0,2,3,1)
                # grid_img = make_grid(torch.unsqueeze(mask,1), nrow=4, normalize=False,scale_each=True)
                # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
                # grid_img = grid_img[:,:,::-1]
                # cv2.imwrite('./test_nan/mask%04d.png'%(idx),grid_img*255)
                stylerig_photo_loss_v = photo_loss(rendered_img[:, :, :, :3]*255, image_tmp.detach(), mask>0)
                image_percept = image * torch.unsqueeze(mask>0, 3).repeat(1,1,1,3).permute(0,3,1,2)
                image_percept = image_percept*2 - 1
                rendered_percept = (rendered_img[:, :, :, :3]*2-1).permute(0,3,1,2)
                perceptual_loss_v = torch.mean(PerceptLoss( image_percept,  rendered_percept))
                ######################################
                sym_gradleftmask = torch.unsqueeze(sym_gradleftmask,3).repeat(1,1,1,3)
                x = {'coeff': sym_coeff,
                     'uvmap': sym_gradleftmask,
                     'uv_displace_map': uv_displace_map,
                     'uv_coord': uv_coord,
                     'uv_coord_2': uv_coord_2,
                     'need_illu': False}
                weighted_mask, _, _, _, _, _ = pure_3dmm(x)
                weighted_mask = weighted_mask[:,:,:,0].detach()
                x = {'coeff': sym_coeff.detach(),
                     'uvmap': uvmap,
                     'uv_displace_map': uv_displace_map,
                     'uv_coord': uv_coord,
                     'uv_coord_2': uv_coord_2}
                sym_rendered_img, _, _, _, _, _ = pure_3dmm(x)
                sym_recon_image = sym_rendered_img[:,:,:,:3].permute(0,3,1,2)
                sym_fake_pred = discriminator(sym_recon_image*2-1)
                sym_g_loss = g_nonsaturating_loss(sym_fake_pred)
                # grid_img = make_grid(torch.unsqueeze(weighted_mask,1), nrow=4, normalize=False,scale_each=True)
                # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
                # cv2.imwrite('./testing/weighted_mask%04d.png'%(idx),grid_img*255)
                sym_stylerig_photo_loss_v = photo_loss(sym_rendered_img[:, :, :, :3]*255, sym_image_255.detach(), weighted_mask)
                sym_mask = sym_rendered_img[:, :, :, 3].detach()
                sym_image_percept = sym_image * torch.unsqueeze(sym_mask>0, 3).repeat(1,1,1,3).permute(0,3,1,2)
                sym_image_percept = sym_image_percept*2-1
                sym_rendered_percept = (sym_rendered_img[:, :, :, :3]*2-1).permute(0,3,1,2)
                sym_perceptual_loss_v = torch.mean(PerceptLoss( sym_image_percept,  sym_rendered_percept))
                #####################################
                loss = 0.75*sym_g_loss  + g_loss + args.photo_weight*(stylerig_photo_loss_v + 0.75*sym_stylerig_photo_loss_v + 0.2*perceptual_loss_v + 0.2*0.75*sym_perceptual_loss_v)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_g_loss += g_loss.item()
        epoch_d_loss = running_d_loss / len(dataloaders) * args.n_critic_d
        epoch_g_loss = running_g_loss / len(dataloaders) * args.n_critic
        epoch_loss = running_loss / len(dataloaders) * args.n_critic
        print("Epoch: %02d, Loss: %f, G_loss: %f, D Loss: %f"%(epoch, epoch_loss, epoch_g_loss, epoch_d_loss))
        towrite = "Epoch: %02d, Loss: %f, G_loss: %f, D Loss: %f"%(epoch, epoch_loss, epoch_g_loss, epoch_d_loss)
        with open(path.join(args.output_dir, 'stats.txt'), 'a') as f:
            f.writelines(towrite + '\n')
        uvmap, uv_displace_map = g_ema(check_data, truncation=1, truncation_latent=None, input_is_Wplus=True, return_uv=True)
        uvmap = torch.tanh(uvmap)
        img = make_grid(uvmap, nrow=4, normalize=True, range=(-1,1),scale_each=True)
        img = img.permute(1,2,0).detach().cpu().numpy()
        img = img[:,:,::-1]
        cv2.imwrite(path.join(args.output_dir, 'uv%02d.png'%(epoch)), img*255)
        uvmap = (uvmap +1)/2
        uvmap = uvmap.permute(0,2,3,1)
        x = {'coeff': check_coeff,
             'uvmap': uvmap,
             'uv_displace_map': uv_displace_map,
             'uv_coord': torch.unsqueeze(uv_coord[0],0).repeat(16,1,1),
             'uv_coord_2': torch.unsqueeze(uv_coord_2[0],0).repeat(16,1,1,1)}
        rendered_img, _, _, _, _, _ = pure_3dmm(x)
        out_img = rendered_img[:,:,:,:3]
        out_img = out_img.permute(0,3,1,2)
        img = make_grid(out_img, nrow=4, normalize=True, range=(0,1),scale_each=True)
        img = img.permute(1,2,0).detach().cpu().numpy()
        img = img[:,:,::-1]
        cv2.imwrite(path.join(args.output_dir, '3dmm_image%02d.png'%(epoch)), img*255)
        best_model_wts = copy.deepcopy(g_ema.state_dict())
        dis_model_wts = copy.deepcopy(discriminator.state_dict())
        if epoch <= 15:
            torch.save(best_model_wts, path.join(args.output_dir,"param%02d.pkl"%(epoch)))
            torch.save(dis_model_wts, path.join(args.output_dir,"dis_param%02d.pkl"%(epoch)))
    return
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_rate",type=float,default=1e-3)
    parser.add_argument("--num_epochs",type=int,default=25)
    parser.add_argument("--batch_size",type=int,default=12)
    parser.add_argument("--output_dir",type=str,default='./checkpoints_final')
    parser.add_argument("--ckpt",type=str,default="./models/StyleGAN2_ffhq_resol=256_550000.pt",help="path to the model checkpoint")
    parser.add_argument("--channel_multiplier",type=int,default=2,help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--latent",type=int,default=512)
    parser.add_argument("--n_mlp",type=int,default=8)
    parser.add_argument("--FACE_MODEL_PATH",type=str,default='./BFM/BFM_model_front.mat')
    parser.add_argument("--load_model",type=str,default='./checkpoints/sep_deeper/param22.pkl')
    parser.add_argument("--n_critic",type=int,default=1)
    parser.add_argument("--n_critic_d",type=int,default=5)
    parser.add_argument("--photo_weight",type=float,default=10)
    parser.add_argument("--d_reg_every",type=int,default=16)
    parser.add_argument("--g_reg_every",type=int,default=4)
    return parser.parse_args()
def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset = StyleCodeImage3DMMParamsPoseDirDataset('./pkls', clean=True)
    dataloaders = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    g_ema = Generator3D(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).cuda()
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
    discriminator = Discriminator3D().cuda()
    discriminator.load_state_dict(checkpoint['d'], strict=False)
    optimizer = optim.Adam(g_ema.parameters(),lr=args.l_rate)
    optimizer_d = optim.Adam(discriminator.parameters(),lr=args.l_rate)
    facemodel = loadmat(args.FACE_MODEL_PATH)
    pure_3dmm = Pure_3DMM_UV(facemodel).cuda()
    # latent_to_posedirection_scalar = latent_to_Multi_PoseDirection_Scalar().cuda()
    # latent_to_posedirection_scalar.load_state_dict(torch.load('./checkpoints_latent2posescalar-new_multi-pose/param.pkl'))
    latent_to_posedirection_scalar = StylecodeToPoseDirectionScalarMLP().cuda()
    latent_to_posedirection_scalar.load_state_dict(torch.load('./checkpoints_latent2posescalar-new-sym/param.pkl'))
    g_ema_orig = Generator_orig(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).cuda()
    g_ema_orig.load_state_dict(checkpoint['g_ema'], strict=False)
    model = StylecodeTo3DMMCoeffMLP().cuda()
    model.load_state_dict(torch.load(args.load_model))
    train(args, dataloaders, g_ema, pure_3dmm, discriminator, optimizer, optimizer_d, latent_to_posedirection_scalar, g_ema_orig, model)
if __name__ =='__main__':
    main()
