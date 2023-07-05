import os, copy

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.io
import lpips

from utils.options import Option
from utils.base_trainer import GANBaseTrainer
from model.face_renderer import FaceRenderer
from model.stylegan2 import Generator2D
from model.stylegan2_texture import Generator3D, Discriminator3D, StylecodeToPoseDirectionScalarMLP, StylecodeTo3DMMCoeffMLP
from utils.losses import *
from utils.utility import output_grid_img_from_tensor, ensure_dir_exist
from dataloader.dataset import StyleCodeImage3DMMParamsPoseDirDataset
from utils.saver import ConstantCheckpointSaver
from trainer_3dcoeff import BestFirst3DCoeffCheckpointSaver
from trainer_pose_scalar import BestFirstPoseScalarCheckpointSaver

debug_output = False


# TODO testing
class Trainer(GANBaseTrainer):
    def init_fn(self):
        ensure_dir_exist(self.options.ckpt_output_dir)

        self.dataset = StyleCodeImage3DMMParamsPoseDirDataset(self.options.dataset_dir, clean=True)
        self.data_loaders = DataLoader(self.dataset, batch_size=self.options.batch_size, shuffle=True, drop_last=True)
        self.g_ema = Generator3D(self.options.size, self.options.latent, self.options.n_mlp).to(self.device)
        self.discriminator = Discriminator3D().to(self.device)

        # 從2d開始繼續train
        checkpoint = torch.load(self.options.generator_2d_ckpt)
        self.g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
        self.discriminator.load_state_dict(checkpoint['d'], strict=False)
        self.optimizer_g = torch.optim.Adam(self.g_ema.parameters(), lr=self.options.l_rate)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.options.l_rate)
        facemodel = scipy.io.loadmat(self.options.FACE_MODEL_PATH)
        self.pure_3dmm_model = FaceRenderer(facemodel).to(self.device)

        self.stylecode_to_scalar_model = StylecodeToPoseDirectionScalarMLP().to(self.device)
        stylecode_to_scalar_saver = BestFirstPoseScalarCheckpointSaver(self.options.stylecode_pose_scalar_mlp_ckpt)
        if stylecode_to_scalar_saver.checkpoint != None:
            stylecode_to_scalar_saver.load_checkpoint(
                models={"Stylecode to Pose Scalar": self.stylecode_to_scalar_model},
                optimizers={})
        self.g_ema_2d = Generator2D(self.options.size, self.options.latent, self.options.n_mlp).to(self.device)
        checkpoint = torch.load(self.options.generator_2d_ckpt)
        self.g_ema_2d.load_state_dict(checkpoint['g_ema'], strict=False)
        self.stylecode_to_3dmm_coeff_model = StylecodeTo3DMMCoeffMLP().to(self.device)
        stylecode_to_3dmm_saver = BestFirst3DCoeffCheckpointSaver(self.options.stylecode_3dmm_coeff_mlp_ckpt)
        if stylecode_to_3dmm_saver.checkpoint != None:
            stylecode_to_3dmm_saver.load_checkpoint(
                models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model},
                optimizers={})

        # Debug data
        self.valid_data = torch.load(self.options.debug_stylecode)[0:16].to(self.device)
        self.valid_coeff = torch.load(self.options.debug_3dmm_coeffs)[0:16].to(self.device)

        self.saver = DispGANConstantCheckpointSaver(self.options.ckpt_output_dir)
        if self.saver.checkpoint != None:
            additional_param = self.saver.load_checkpoint(
                models={"Generator": self.g_ema, "Discriminator": self.discriminator},
                optimizers={"Generator Optimizer": self.optimizer_g, "Discriminator Optimizer": self.optimizer_d})

            self.epoch = additional_param['epoch']
            self.batch_idx = additional_param['batch_idx']
            self.options.l_rate = additional_param['l_rate']
            self.options.num_epochs = additional_param['num_epochs']
            self.options.batch_size = additional_param['batch_size']
            self.options.n_critic = additional_param['n_critic']
            self.options.n_critic_d = additional_param['n_critic_d']
            self.options.photo_weight = additional_param['photo_weight']
            self.options.d_reg_every = additional_param['d_reg_every']
            self.options.g_reg_every = additional_param['g_reg_every']
            self.d_cnt = additional_param['d_cnt']
            self.g_cnt = additional_param['g_cnt']
        else:
            self.d_cnt = 0
            self.g_cnt = 0

        self.pure_3dmm_model.eval()
        self.g_ema_2d.eval()
        self.stylecode_to_3dmm_coeff_model.eval()
        self.stylecode_to_scalar_model.eval()
        self.g_ema.train()
        self.discriminator.train()
        self.uv_coord_2 = torch.tensor(scipy.io.loadmat(self.options.FACE_UV_MODEL_PATH)['UV_refine'])
        self.uv_coord_2 = self.process_uv(self.uv_coord_2)
        self.uv_coord_2 = self.uv_coord_2[:, :2] / 255.
        self.uv_coord_2 = self.uv_coord_2 * 2 - 1
        self.uv_coord_2 = torch.unsqueeze(torch.tensor(self.uv_coord_2), 0).type('torch.DoubleTensor').to(self.device)
        self.uv_coord_2 = self.uv_coord_2.repeat(self.options.batch_size, 1, 1)
        self.uv_coord_2 = torch.unsqueeze(self.uv_coord_2, 1)

        # self.uvmask (batch_size, 256, 256)
        self.uvmask = cv2.imread(os.path.join(self.options.dataset_dir, 'mask.png'), 0)
        self.uvmask = self.uvmask > 0
        self.uvmask = torch.tensor(self.uvmask).to(self.device)
        self.uvmask = torch.unsqueeze(self.uvmask, 0).repeat(self.options.batch_size, 1, 1)
        self.uvmask = self.uvmask.detach()

        self.PerceptLoss = lpips.LPIPS(net='vgg').to(self.device)

    def train_step(self, idx, input_batch, stage):
        style_code = input_batch[0].to(self.device)
        image = input_batch[1].to(self.device)
        image = image.float()
        coeff = input_batch[2].to(self.device)
        pose_direction = input_batch[3].to(self.device)
        pose_direction = pose_direction.detach()

        # [running_loss, running_d_loss, running_g_loss]
        running_losses = [0.0] * 3
        #################################################
        ########  Get angles with symmetric view
        ##################################################
        scalar = self.stylecode_to_scalar_model(torch.flatten(style_code, start_dim=1)).view(-1, 1, 1)
        #####################################################
        symmetric_style_code = style_code + scalar * pose_direction
        sym_image, _ = self.g_ema_2d(symmetric_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
        sym_image = torch.clamp(sym_image, -1, 1)
        sym_image = ((sym_image + 1) / 2).detach()
        sym_image_255 = sym_image.permute(0, 2, 3, 1) * 255
        sym_coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(symmetric_style_code, start_dim=1))
        ####################################
        sym_gradmask = self.get_gradmask(sym_coeff.detach()).detach()
        gradmask = self.get_gradmask(coeff.detach()).detach()  # (b, 256, 256) 0~1
        sym_gradleftmask = (sym_gradmask - (sym_gradmask > 0) * (gradmask > 0) * 1.)
        # simplify to this.
        # sym_gradleftmask = (self.uvmask > 0) * (~(sym_gradleftmask > 0)) * 0.5 + sym_gradleftmask
        sym_gradleftmask = (self.uvmask > 0) * (~((self.uvmask > 0) * (sym_gradleftmask > 0))) * 0.5 + sym_gradleftmask
        # grid_img = make_grid(torch.unsqueeze(sym_gradleftmask,1), nrow=4, normalize=False,scale_each=True)
        # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('./testing/mask%04d.png'%(idx),grid_img*255)
        if idx % self.options.n_critic_d == 0:
            self.d_cnt += 1
            self.requires_grad(self.g_ema, False)
            self.requires_grad(self.discriminator, True)
            self.optimizer_d.zero_grad()
            # uvmap = g_ema(style_code)
            uvmap, uv_displace_map = self.g_ema(style_code, truncation=1, truncation_latent=None, input_is_Wplus=True,
                                                return_uv=True)
            # FIXME uvmap should be refactor to some more related term..
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            x = {'coeff': coeff,
                 'uvmap': uvmap,
                 'uv_displace_map': uv_displace_map,
                 'uv_coord_2': self.uv_coord_2}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_model(x)
            mask = rendered_img[:, :, :, 3].detach()
            mask = (mask > 0)
            mask = mask.unsqueeze(3)
            mask = mask.repeat((1, 1, 1, 3))
            mask = mask.permute(0, 3, 1, 2)
            recon_image = rendered_img[:, :, :, :3].permute(0, 3, 1, 2)
            c = image * mask
            c = (c * 2) - 1
            real_pred = self.discriminator(c)
            fake_pred = self.discriminator(recon_image * 2 - 1)  # .detach())
            d_loss = d_logistic_loss(real_pred, fake_pred)
            ###################################
            x = {'coeff': sym_coeff.detach(),
                 'uvmap': uvmap,
                 'uv_displace_map': uv_displace_map,
                 'uv_coord_2': self.uv_coord_2}
            sym_rendered_img, _, _, _, _, _ = self.pure_3dmm_model(x)
            sym_recon_image = sym_rendered_img[:, :, :, :3].permute(0, 3, 1, 2)
            sym_mask = sym_rendered_img[:, :, :, 3].detach()
            sym_mask = (sym_mask > 0)
            sym_mask = sym_mask.unsqueeze(3)
            sym_mask = sym_mask.repeat((1, 1, 1, 3))
            sym_mask = sym_mask.permute(0, 3, 1, 2)
            sym_tmp_img = sym_image * sym_mask
            # grid_img = make_grid(sym_tmp_img, nrow=4, normalize=False,scale_each=True)
            # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
            # grid_img = grid_img[:,:,::-1]
            # cv2.imwrite('./testing/sym_image-mask%04d.png'%(idx),grid_img*255)
            sym_real_pred = self.discriminator(sym_tmp_img * 2 - 1)
            sym_fake_pred = self.discriminator(sym_recon_image * 2 - 1)
            sym_d_loss = d_logistic_loss(sym_real_pred, sym_fake_pred)
            ##################################
            self.discriminator.zero_grad()
            (0.75 * sym_d_loss + d_loss).backward()
            self.optimizer_d.step()
            running_losses[1] += (d_loss.item() + sym_d_loss.item())
            ####
            if self.d_cnt % self.options.d_reg_every == 0:
                # print("YO")
                self.optimizer_d.zero_grad()
                c2 = image * mask
                image_masked = (c2 * 2) - 1
                image_masked.requires_grad = True
                real_pred = self.discriminator(image_masked)
                r1_loss = d_r1_loss(real_pred, image_masked)
                #######3
                c3 = sym_image * sym_mask
                sym_image_masked = (c3 * 2) - 1
                sym_image_masked.requires_grad = True
                sym_real_pred = self.discriminator(sym_image_masked)
                sym_r1_loss = d_r1_loss(sym_real_pred, sym_image_masked)
                #######
                self.discriminator.zero_grad()
                ((10 / 2 * r1_loss * self.options.d_reg_every + 0 * real_pred[0]) + 0.75 * (
                        10 / 2 * sym_r1_loss * self.options.d_reg_every + 0 * sym_real_pred[0])).backward()
                self.optimizer_d.step()
        if idx % self.options.n_critic == 0:
            self.g_cnt += 1
            self.requires_grad(self.g_ema, True)
            self.requires_grad(self.discriminator, False)
            self.optimizer_g.zero_grad()
            uvmap, uv_displace_map = self.g_ema(style_code, truncation=1, truncation_latent=None, input_is_Wplus=True,
                                                return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            x = {'coeff': coeff,
                 'uvmap': uvmap,
                 'uv_displace_map': uv_displace_map,
                 'uv_coord_2': self.uv_coord_2}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_model(x)
            mask = rendered_img[:, :, :, 3].detach()
            recon_image = rendered_img[:, :, :, :3].permute(0, 3, 1, 2)
            fake_pred = self.discriminator(recon_image * 2 - 1)
            g_loss = g_nonsaturating_loss(fake_pred)
            image_tmp = (image * 255).permute(0, 2, 3, 1)
            # grid_img = make_grid(torch.unsqueeze(mask,1), nrow=4, normalize=False,scale_each=True)
            # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
            # grid_img = grid_img[:,:,::-1]
            # cv2.imwrite('./test_nan/mask%04d.png'%(idx),grid_img*255)
            stylerig_photo_loss_v = photo_loss(rendered_img[:, :, :, :3] * 255, image_tmp.detach(), mask > 0)
            image_percept = image * torch.unsqueeze(mask > 0, 3).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
            image_percept = image_percept * 2 - 1
            rendered_percept = (rendered_img[:, :, :, :3] * 2 - 1).permute(0, 3, 1, 2)
            perceptual_loss_v = torch.mean(self.PerceptLoss(image_percept, rendered_percept))
            ######################################
            sym_gradleftmask = torch.unsqueeze(sym_gradleftmask, 3).repeat(1, 1, 1, 3)
            x = {'coeff': sym_coeff,
                 'uvmap': sym_gradleftmask,
                 'uv_displace_map': uv_displace_map,
                 'uv_coord_2': self.uv_coord_2,
                 'need_illu': False}
            weighted_mask, _, _, _, _, _ = self.pure_3dmm_model(x)
            weighted_mask = weighted_mask[:, :, :, 0].detach()
            x = {'coeff': sym_coeff.detach(),
                 'uvmap': uvmap,
                 'uv_displace_map': uv_displace_map,
                 'uv_coord_2': self.uv_coord_2}
            sym_rendered_img, _, _, _, _, _ = self.pure_3dmm_model(x)
            sym_recon_image = sym_rendered_img[:, :, :, :3].permute(0, 3, 1, 2)
            sym_fake_pred = self.discriminator(sym_recon_image * 2 - 1)
            sym_g_loss = g_nonsaturating_loss(sym_fake_pred)
            # grid_img = make_grid(torch.unsqueeze(weighted_mask,1), nrow=4, normalize=False,scale_each=True)
            # grid_img = grid_img.permute(1,2,0).detach().cpu().numpy()
            # cv2.imwrite('./testing/weighted_mask%04d.png'%(idx),grid_img*255)
            sym_stylerig_photo_loss_v = photo_loss(sym_rendered_img[:, :, :, :3] * 255, sym_image_255.detach(),
                                                   weighted_mask)
            sym_mask = sym_rendered_img[:, :, :, 3].detach()
            sym_image_percept = sym_image * torch.unsqueeze(sym_mask > 0, 3).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
            sym_image_percept = sym_image_percept * 2 - 1
            sym_rendered_percept = (sym_rendered_img[:, :, :, :3] * 2 - 1).permute(0, 3, 1, 2)
            sym_perceptual_loss_v = torch.mean(self.PerceptLoss(sym_image_percept, sym_rendered_percept))
            #####################################
            loss = 0.75 * sym_g_loss + g_loss + self.options.photo_weight * (
                    stylerig_photo_loss_v + 0.75 * sym_stylerig_photo_loss_v + 0.2 * perceptual_loss_v + 0.2 * 0.75 * sym_perceptual_loss_v)
            loss.backward()
            self.optimizer_g.step()
            running_losses[0] += loss.item()
            running_losses[2] += g_loss.item()
        return running_losses

    def post_train_step(self, epoch, epoch_loss):
        super(Trainer, self).post_train_step(epoch, epoch_loss)
        uvmap, uv_displace_map = self.g_ema(self.valid_data, truncation=1, truncation_latent=None, input_is_Wplus=True,
                                            return_uv=True)
        uvmap = torch.tanh(uvmap)
        debug_output_dir = os.path.join(self.options.debug_output_dir, "3d")
        ensure_dir_exist(debug_output_dir)
        output_grid_img_from_tensor(uvmap, os.path.join(debug_output_dir, 'uv%02d.png' % (epoch)))

        uvmap = (uvmap + 1) / 2
        uvmap = uvmap.permute(0, 2, 3, 1)
        x = {'coeff': self.valid_coeff,
             'uvmap': uvmap,
             'uv_displace_map': uv_displace_map,
             'uv_coord_2': torch.unsqueeze(self.uv_coord_2[0], 0).repeat(16, 1, 1, 1)}
        rendered_img, _, _, _, _, _ = self.pure_3dmm_model(x)
        out_img = rendered_img[:, :, :, :3]
        out_img = out_img.permute(0, 3, 1, 2)
        debug_output_dir = os.path.join(self.options.debug_output_dir, "3d")
        ensure_dir_exist(debug_output_dir)
        output_grid_img_from_tensor(out_img, os.path.join(debug_output_dir, '3dmm_image%02d.png' % (epoch)))

        self.saver.save_checkpoint(models={"Generator": self.g_ema, "Discriminator": self.discriminator},
                                   optimizers={"Generator Optimizer": self.optimizer_g,
                                               "Discriminator Optimizer": self.optimizer_d},
                                   trainer=self)

        # best_model_wts = copy.deepcopy(self.g_ema.state_dict())
        # dis_model_wts = copy.deepcopy(self.discriminator.state_dict())
        # if epoch <= 15:
        #     torch.save(best_model_wts, os.path.join(self.options.ckpt_output_dir, "param%02d.pkl" % (epoch)))
        #     torch.save(dis_model_wts, os.path.join(self.options.ckpt_output_dir, "dis_param%02d.pkl" % (epoch)))

    @staticmethod
    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
        return uv_coords

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def get_gradmask(self, sym_coeff):
        grad_uvmap = Variable(torch.ones(self.options.batch_size, 3, 256, 256) * 100, requires_grad=False).to(
            self.device)
        grad_uvmap = grad_uvmap.permute(0, 2, 3, 1)
        grad_uv_displace_map = Variable(torch.ones(self.options.batch_size, 3, 256, 256) * 0, requires_grad=True).to(
            self.device)
        x = {'coeff': sym_coeff.detach(),
             'uvmap': grad_uvmap,
             'uv_displace_map': grad_uv_displace_map,
             'uv_coord_2': self.uv_coord_2}
        grad_rendered_img, _, _, _, _, _ = self.pure_3dmm_model(x)
        grad_uv_displace_map.retain_grad()
        (torch.sum(grad_rendered_img[:, :, :, :3] * 255)).backward()
        gradmask = torch.sum(torch.abs(grad_uv_displace_map.grad), dim=1)  # (b, 256, 256)
        gradmask = (gradmask != 0) * 1.
        # gradmask = torch.unsqueeze(gradmask, 1)
        # gradmask = make_grid(gradmask, nrow=4, normalize=False,scale_each=True)
        # gradmask = gradmask.permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('./testing/image%04d.png'%(idx),gradmask*255)
        return gradmask


class DispGANConstantCheckpointSaver(ConstantCheckpointSaver):
    # inheritance
    def get_model_related_attribute(self, trainer):
        return {'l_rate': trainer.options.l_rate, 'num_epochs': trainer.options.num_epochs,
                'batch_size': trainer.options.batch_size, 'n_critic': trainer.options.n_critic,
                'n_critic_d': trainer.options.n_critic_d, 'photo_weight': trainer.options.photo_weight,
                'd_reg_every': trainer.options.d_reg_every, 'g_reg_every': trainer.options.g_reg_every,
                'd_cnt': trainer.d_cnt, 'g_cnt': trainer.g_cnt
                }

    # inheritance
    def get_model_related_attribute_from_checkpoint(self, checkpoint):
        return {'l_rate': checkpoint['l_rate'], 'num_epochs': checkpoint['num_epochs'],
                'batch_size': checkpoint['batch_size'], 'n_critic': checkpoint['n_critic'],
                'n_critic_d': checkpoint['n_critic_d'], 'photo_weight': checkpoint['photo_weight'],
                'd_reg_every': checkpoint['d_reg_every'], 'g_reg_every': checkpoint['g_reg_every'],
                'd_cnt': checkpoint['d_cnt'], 'g_cnt': checkpoint['g_cnt']
                }


if __name__ == '__main__':
    option = Option("Disp GAN")
    args = option.parse_args()
    trainer = Trainer(args)
    trainer.train()

