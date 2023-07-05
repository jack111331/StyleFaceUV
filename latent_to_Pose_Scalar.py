import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import os, copy, cv2
import os.path as path
from dataloader.dataset import StyleCode3DMMParamsDataset
from .network.arch import StylecodeToPoseDirectionScalarMLP, StylecodeTo3DMMCoeffMLP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.losses import l1loss
from .network.arch import Generator2D
from torchvision import transforms
from torchvision.utils import make_grid

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True), #(-1, 1)
    ]
)
###############################################################
#####    StyleCode to Pose Scalar with symmetric angle   #######
###############################################################
def test(args, g_ema, latent_to_poseDirection_scalar):
    #check_data = torch.load('./pkls/tensor-new.pkl')[29148-15:29149].cuda()
    check_data = []
    pkls = torch.load('./pkls/3DMMparam-new.pkl')
    tensors = torch.load('./pkls/tensor-new.pkl')
    cnt = 0
    for i in range(len(pkls)):
        if(abs(pkls[i][225].item())> 0.5):
            check_data.append(tensors[i])
            cnt += 1
        if cnt==16:
            break
    check_data = torch.stack(check_data).cuda()
    img, _ = g_ema(check_data, truncation=1, truncation_latent=None, input_is_Wplus=True)    
    img = make_grid(img, nrow=4, normalize=True, range=(-1,1),scale_each=True)
    img = img.permute(1,2,0).detach().cpu().numpy()
    img = img[:,:,::-1]
    cv2.imwrite(path.join(args.output_dir, 'ori_image.png'), img*255)
    best_loss = 100
    pose_direction = torch.load(path.join('./pkls','pose_direction-new.pkl')).view(1,14,512).type(torch.FloatTensor)
    pose_direction = pose_direction.repeat(args.batch_size,1,1).cuda()
    scalar = latent_to_poseDirection_scalar(torch.flatten(check_data, start_dim=1))
    scalar = scalar.view(-1,1,1)
    img, _ = g_ema(check_data+scalar*(torch.unsqueeze(pose_direction[0],0).repeat(16,1,1)), truncation=1, truncation_latent=None, input_is_Wplus=True)    
    img = make_grid(img, nrow=4, normalize=True, range=(-1,1),scale_each=True)
    img = img.permute(1,2,0).detach().cpu().numpy()
    img = img[:,:,::-1]
    cv2.imwrite(path.join(args.output_dir, 'image%02d.png'%(0)), img*255)
def train(args,dataloaders, model,  optimizer, scheduler, g_ema, latent_to_poseDirection_scalar):
    best_model_wts = copy.deepcopy(latent_to_poseDirection_scalar.state_dict())
    #check_data = torch.unsqueeze(torch.load('./pkls/tensor-new.pkl')[0], 0).cuda()
    check_data = torch.load('./pkls/tensor-new.pkl')[16:32].cuda()
    img, _ = g_ema(check_data, truncation=1, truncation_latent=None, input_is_Wplus=True)    
    img = make_grid(img, nrow=4, normalize=True, range=(-1,1),scale_each=True)
    img = img.permute(1,2,0).detach().cpu().numpy()
    img = img[:,:,::-1]
    cv2.imwrite(path.join(args.output_dir, 'ori_image.png'), img*255)
    best_loss = 100
    g_ema.eval()
    model.eval()
    pose_direction = torch.load(path.join('./pkls','pose_direction-new.pkl')).view(1,14,512).type(torch.FloatTensor)
    pose_direction = pose_direction.repeat(args.batch_size,1,1).cuda()
    target = torch.ones(args.batch_size)*1.0471975512*-1
    target = target.cuda()
    for epoch in range(1,args.num_epochs+1):
        print('Epoch {}/{}'.format(epoch, args.num_epochs ))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                latent_to_poseDirection_scalar.train()
            else:
                latent_to_poseDirection_scalar.eval()
            running_loss = 0.0
            for idx,data in tqdm(enumerate(dataloaders[phase]),total=len(dataloaders[phase])):
                stylecode = data[0].cuda()
                coeffs = data[1].cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    scalar = latent_to_poseDirection_scalar(torch.flatten(stylecode, start_dim=1))
                    scalar = scalar.view(-1,1,1)
                    sym_stylecode = stylecode + scalar * pose_direction
                    sym_coeffs = model(torch.flatten(sym_stylecode, start_dim=1))
                    optimizer.zero_grad()
                    #print(coeffs[:,225].size())
                    loss = l1loss(sym_coeffs[:,225],coeffs[:,225]*-1)
                    #loss = l1loss(sym_coeffs[:,225], target)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders[phase])
            print("Phase: %s, Epoch: %02d, Loss: %f"%(phase, epoch, epoch_loss))
            if(phase=='val' and epoch_loss < best_loss):
                print("Got New Best !")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(latent_to_poseDirection_scalar.state_dict())
                torch.save(best_model_wts, path.join(args.output_dir,"param.pkl"))
                scalar = latent_to_poseDirection_scalar(torch.flatten(check_data, start_dim=1))
                scalar = scalar.view(-1,1,1)
                img, _ = g_ema(check_data+scalar*(torch.unsqueeze(pose_direction[0],0).repeat(16,1,1)), truncation=1, truncation_latent=None, input_is_Wplus=True)    
                img = make_grid(img, nrow=4, normalize=True, range=(-1,1),scale_each=True)
                img = img.permute(1,2,0).detach().cpu().numpy()
                img = img[:,:,::-1]
                cv2.imwrite(path.join(args.output_dir, 'image%02d.png'%(epoch)), img*255)
                #print(img.size())
                # img = img.permute(0,2,3,1)
                # img = ((img+1)/2).detach().cpu().numpy()[0]
                # img = img[:,:,::-1]
                # cv2.imwrite(path.join(args.output_dir, 'image%02d.png'%(epoch)), img*255)
        if scheduler is not None:
            scheduler.step()
    return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_rate",type=float,default=1e-3)
    parser.add_argument("--num_epochs",type=int,default=25)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--output_dir",type=str,default='./checkpoints_latent2posescalar-new-sym')
    parser.add_argument("--ckpt",type=str,default="./models/StyleGAN2_ffhq_resol=256_550000.pt",help="path to the model checkpoint")
    parser.add_argument("--channel_multiplier",type=int,default=2,help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--latent",type=int,default=512)
    parser.add_argument("--n_mlp",type=int,default=8)
    parser.add_argument("--load_model",type=str,default='./checkpoints/sep_deeper/param22.pkl')
    parser.add_argument("--FACE_MODEL_PATH",type=str,default='./BFM/BFM_model_front.mat')
    parser.add_argument("--mode", type=str, default='train')
    return parser.parse_args()
def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    g_ema = Generator2D(size=256, style_dim=512, n_mlp=8).cuda()
    g_ema.load_state_dict(torch.load('./models/StyleGAN2_ffhq_resol=256_550000.pt')['g_ema'], strict=False)
    latent_to_poseDirection_scalar = StylecodeToPoseDirectionScalarMLP().cuda()
    if args.mode == 'train':  
        dataset = StyleCode3DMMParamsDataset('./pkls')
        datasets = {'train':dataset[:int(len(dataset)*0.9)],'val':dataset[int(len(dataset)*0.9):]}
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, drop_last=True) for x in ['train','val']}
        model = StylecodeTo3DMMCoeffMLP().cuda()
        model.load_state_dict(torch.load(args.load_model))
        optimizer = optim.Adam(latent_to_poseDirection_scalar.parameters(),lr=args.l_rate)
        scheduler = MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
        train(args,dataloaders, model, optimizer, scheduler, g_ema, latent_to_poseDirection_scalar)       
    elif args.mode=='test':
        latent_to_poseDirection_scalar.load_state_dict(torch.load('./checkpoints_latent2posescalar-new-sym/param.pkl')) # "Load ur checkpoints here"
        test(args, g_ema, latent_to_poseDirection_scalar)
      
if __name__ =='__main__':
    main()