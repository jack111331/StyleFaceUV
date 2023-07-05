import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os, copy, cv2, math
import os.path as path
from dataloader.dataset import StyleCode3DMMParamsDataset
from .utils.options import Option
from .utils.utility import output_grid_img_from_tensor
from .network.arch import StylecodeToMultiPoseDirectionScalarMLP, StylecodeTo3DMMCoeffMLP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.losses import l1loss
from .network.arch import Generator2D
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True), #(-1, 1)
    ]
)
###############################################################
#####    StyleCode to Pose Scalar with selected angle   #######
###############################################################
def test(args, g_ema, latent_to_poseDirection_scalar):
    pose_direction = torch.load(path.join('./pkls','pose_direction-new.pkl')).view(1,14,512).type(torch.FloatTensor)
    pose_direction = pose_direction.cuda()
    nums = np.arange(5).tolist()
    for ind,i in tqdm(enumerate(nums),total=len(nums)):
        latent = torch.randn(1, 512).cuda()
        style_code = g_ema.get_latent_Wplus(latent)
        for item in [0,15,-15,30,-30,45,-45,60,-60]:
            degree = torch.tensor(item * math.pi / 180).cuda()
            degree = degree.view(1,1)
            scalar = latent_to_poseDirection_scalar(torch.flatten(style_code, start_dim=1), degree)
            scalar = scalar.view(-1,1,1)
            sym_stylecode = style_code+scalar*(torch.unsqueeze(pose_direction[0],0).repeat(1,1,1))
            img, _ = g_ema(sym_stylecode, truncation=1, truncation_latent=None, input_is_Wplus=True)
            img = torch.clamp(img, -1, 1)
            img = img.permute(0,2,3,1)
            img = (img+1)/2
            img = img.detach().cpu().numpy()[0]
            img = img[:,:,::-1]
            cv2.imwrite(path.join(args.output_dir,'test%04d_%d.png'%(i,item)), img*255 )
def train(args,dataloaders, model,  optimizer, scheduler, g_ema, latent_to_poseDirection_scalar):
    check_data = torch.load('./pkls/tensor-new.pkl')[16:32].cuda()
    img_tensor, _ = g_ema(check_data, truncation=1, truncation_latent=None, input_is_Wplus=True)
    output_grid_img_from_tensor(img_tensor, path.join(args.output_dir, 'ori_image.png'))
    best_loss = 100
    g_ema.eval()
    model.eval()
    # pose_normal_direction 是 讓stylecode往這個方向前進會讓 Pose Semantic 從向左變向右, reference from InterfaceGAN
    pose_normal_direction = torch.load(path.join('./pkls','pose_direction-new.pkl')).view(1,14,512).type(torch.FloatTensor)
    pose_normal_direction = pose_normal_direction.repeat(args.batch_size,1,1).cuda()
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
                # 改deg2rad
                sample_radian = torch.deg2rad(torch.FloatTensor(args.batch_size).uniform_(-30, 30)) # it's uniform sampled between [-30 degrees, 30 degrees]
                sample_radian = sample_radian.cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    scalar = latent_to_poseDirection_scalar(torch.flatten(stylecode, start_dim=1), torch.unsqueeze(sample_radian, 1))
                    scalar = scalar.view(-1,1,1)
                    sym_stylecode = stylecode + scalar * pose_normal_direction
                    sym_3dmm_coeffs = model(torch.flatten(sym_stylecode, start_dim=1))
                    optimizer.zero_grad()
                    loss = l1loss(sym_3dmm_coeffs[:,225], sample_radian)
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
                scalar = latent_to_poseDirection_scalar(torch.flatten(check_data, start_dim=1), torch.zeros(16,1).cuda())
                scalar = scalar.view(-1,1,1)
                img_tensor, _ = g_ema(check_data+scalar*(torch.unsqueeze(pose_normal_direction[0],0).repeat(16,1,1)), truncation=1, truncation_latent=None, input_is_Wplus=True)
                output_grid_img_from_tensor(img_tensor, path.join(args.output_dir, 'image%02d.png' % (epoch)))
        if scheduler is not None:
            scheduler.step()
    return

if __name__ == '__main__':
    option = Option("Style Code To Pose Direction MLP")
    args = option.parse_args()
    if not os.path.exists(args.ckpt_output_dir):
        os.makedirs(args.ckpt_output_dir)
    g_ema = Generator2D(size=args.size, style_dim=args.latent, n_mlp=args.n_mlp).cuda()
    g_ema.load_state_dict(torch.load(args.generator_2d_ckpt)['g_ema'], strict=False)
    latent_to_poseDirection_scalar = StylecodeToMultiPoseDirectionScalarMLP().cuda()

    if args.mode == 'train':
        dataset = StyleCode3DMMParamsDataset('./pkls')
        datasets = {'train':dataset[:int(len(dataset)*0.9)],'val':dataset[int(len(dataset)*0.9):]}
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, drop_last=True) for x in ['train','val']}
        model = StylecodeTo3DMMCoeffMLP().cuda()
        model.load_state_dict(torch.load(args.load_model))
        optimizer = optim.Adam(latent_to_poseDirection_scalar.parameters(),lr=args.l_rate)
        scheduler = MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
        train(args,dataloaders, model, optimizer, scheduler, g_ema, latent_to_poseDirection_scalar)
    # elif args.mode=='test':
    #     # FIXME test 移到一個統一的地方
    #     latent_to_poseDirection_scalar.load_state_dict(torch.load('./checkpoints_latent2posescalar-new_multi-pose/param.pkl'))
    #     test(args, g_ema, latent_to_poseDirection_scalar)