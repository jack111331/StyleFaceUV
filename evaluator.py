import os, copy

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import scipy.io
import lpips
import math
from tqdm import tqdm
from imageio import mimwrite
from trainer_3d import DispGANConstantCheckpointSaver
from trainer_3dcoeff import BestFirst3DCoeffCheckpointSaver
from trainer_pose_scalar import BestFirstPoseScalarCheckpointSaver
from trainer_multi_pose_scalar import BestFirstMultiPoseScalarCheckpointSaver


from model.face_renderer import FaceRenderer
from model.stylegan2 import Generator2D
from model.stylegan2_texture import Generator3D, StylecodeToPoseDirectionScalarMLP, StylecodeToMultiPoseDirectionScalarMLP, StylecodeTo3DMMCoeffMLP
from utils.losses import *
from utils.utility import output_grid_img_from_tensor, ensure_dir_exist, output_img_from_tensor, output_3dmm_img_from_tensor


debug_output = False

class Evaluator(object):
    def __init__(self, device, options):
        super(Evaluator, self).__init__()

        self.device = device
        self.options = options

        self.stylecode_to_3dmm_coeff_model = StylecodeTo3DMMCoeffMLP().to(self.device)
        stylecode_to_3dmm_saver = BestFirst3DCoeffCheckpointSaver(self.options.stylecode_3dmm_coeff_mlp_ckpt)
        if stylecode_to_3dmm_saver.checkpoint != None:
            stylecode_to_3dmm_saver.load_checkpoint(
                models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model},
                optimizers={})

        facemodel = scipy.io.loadmat(self.options.FACE_MODEL_PATH)
        self.pure_3dmm_uv_model = FaceRenderer(facemodel).to(self.device)
        self.pure_3dmm_uv_model.eval()

        self.stylecode_to_scalar_model = StylecodeToPoseDirectionScalarMLP().to(self.device)
        stylecode_to_scalar_saver = BestFirstPoseScalarCheckpointSaver(self.options.stylecode_pose_scalar_mlp_ckpt)
        if stylecode_to_scalar_saver.checkpoint != None:
            stylecode_to_scalar_saver.load_checkpoint(
                models={"Stylecode to Pose Scalar": self.stylecode_to_scalar_model},
                optimizers={})

        self.stylecode_to_multi_scalar_model = StylecodeToMultiPoseDirectionScalarMLP().to(self.device)
        stylecode_to_multi_scalar_saver = BestFirstMultiPoseScalarCheckpointSaver(self.options.stylecode_multi_pose_scalar_mlp_ckpt)
        if stylecode_to_multi_scalar_saver.checkpoint != None:
            stylecode_to_multi_scalar_saver.load_checkpoint(
                models={"Stylecode to Multi Pose Scalar": self.stylecode_to_multi_scalar_model},
                optimizers={})

        self.g_ema_2d = Generator2D(self.options.size, self.options.latent, self.options.n_mlp).to(self.device)
        checkpoint = torch.load(self.options.generator_2d_ckpt)
        self.g_ema_2d.load_state_dict(checkpoint['g_ema'], strict=False)
        
        self.g_ema = Generator3D(self.options.size, self.options.latent, self.options.n_mlp).to(self.device)
        disp_gan_saver = DispGANConstantCheckpointSaver(self.options.generator_3d_ckpt)
        if disp_gan_saver.checkpoint != None:
            disp_gan_saver.load_checkpoint(
                models={"Generator": self.g_ema},
                optimizers={})


        self.uv_coord_2 = torch.tensor(scipy.io.loadmat('data/BFM_UV_refined.mat')['UV_refine'])
        self.uv_coord_2 = self.process_uv(self.uv_coord_2)
        self.uv_coord_2 = self.uv_coord_2[:, :2] / 255.
        self.uv_coord_2 = self.uv_coord_2 * 2 - 1
        self.uv_coord_2 = torch.unsqueeze(self.uv_coord_2, 0).type('torch.DoubleTensor').to(self.device)
        self.uv_coord_2 = self.uv_coord_2.repeat(1, 1, 1)
        self.uv_coord_2 = torch.unsqueeze(self.uv_coord_2, 1)

        # self.uvmask (batch_size, 256, 256)
        self.uvmask = cv2.imread(os.path.join(self.options.dataset_dir, 'mask.png'), 0)
        self.uvmask = self.uvmask > 0
        self.uvmask = torch.tensor(self.uvmask).to(self.device)
        self.uvmask = torch.unsqueeze(self.uvmask, 0)
        self.uvmask = self.uvmask.detach()

        self.pose_direction = torch.load(os.path.join(self.options.dataset_dir, 'pose_direction-new.pkl')).view(14, 512).type(torch.FloatTensor).to(self.device)


    @staticmethod
    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        return uv_coords

    def get_gradmask(self, sym_coeff):
        grad_uvmap = Variable(torch.ones(1, 3, 256, 256) * 100, requires_grad=False).to(
            self.device)
        grad_uvmap = grad_uvmap.permute(0, 2, 3, 1)
        grad_uv_displace_map = Variable(torch.ones(1, 3, 256, 256) * 0, requires_grad=True).to(
            self.device)
        # print(grad_uvmap, grad_uv_displace_map)
        x = {'coeff': sym_coeff.detach(),
             'uvmap': grad_uvmap,
             'uv_displace_map': grad_uv_displace_map,
             'uv_coord_2': self.uv_coord_2}
        grad_rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
        grad_uv_displace_map.retain_grad()
        (torch.sum(grad_rendered_img[:, :, :, :3] * 255)).backward()
        gradmask = torch.sum(torch.abs(grad_uv_displace_map.grad), dim=1)  # (b, 256, 256)
        gradmask = (gradmask != 0) * 1.
        # gradmask = torch.unsqueeze(gradmask, 1)
        # gradmask = make_grid(gradmask, nrow=4, normalize=False,scale_each=True)
        # gradmask = gradmask.permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('./testing/image%04d.png'%(idx),gradmask*255)
        return gradmask

    def test_pose_scalar(self):
        check_data = []
        pkls = torch.load(self.options.debug_3dmm_coeffs)
        tensors = torch.load(self.options.debug_stylecode)
        cnt = 0
        for i in range(len(pkls)):
            if (abs(pkls[i][225].item()) > 0.5):
                check_data.append(tensors[i])
                cnt += 1
            if cnt == 16:
                break
        check_data = torch.stack(check_data).to(self.device)
        img, _ = self.g_ema_2d(check_data, truncation=1, truncation_latent=None, input_is_Wplus=True)
        test_output_dir = os.path.join(self.options.test_output_dir, "pose_scalar")
        ensure_dir_exist(test_output_dir)
        output_grid_img_from_tensor(img, os.path.join(test_output_dir, 'ori_image.png'))
        pose_direction = torch.load(self.options.debug_pose_normal_direction).view(1, 14, 512).type(
            torch.FloatTensor)
        pose_direction = pose_direction.repeat(16, 1, 1).to(self.device)
        scalar = self.stylecode_to_scalar_model(torch.flatten(check_data, start_dim=1))
        scalar = scalar.view(-1, 1, 1)
        img, _ = self.g_ema_2d(check_data + scalar * (torch.unsqueeze(pose_direction[0], 0).repeat(16, 1, 1)), truncation=1,
                       truncation_latent=None, input_is_Wplus=True)
        test_output_dir = os.path.join(self.options.test_output_dir, "pose_scalar")
        ensure_dir_exist(test_output_dir)
        output_grid_img_from_tensor(img, os.path.join(test_output_dir, 'image%02d.png' % (0)))

    def test_multi_pose_scalar(self):
        pose_direction = torch.load(os.path.join(self.options.debug_pose_normal_direction)).view(1, 14, 512).type(
            torch.FloatTensor)
        pose_direction = pose_direction.to(self.device)
        nums = np.arange(5).tolist()
        for ind, i in tqdm(enumerate(nums), total=len(nums)):
            latent = torch.randn(1, 512).to(self.device)
            style_code = self.g_ema_2d.get_latent_Wplus(latent)
            for item in [0, 15, -15, 30, -30, 45, -45, 60, -60]:
                degree = torch.tensor(item * math.pi / 180).to(self.device)
                degree = degree.view(1, 1)
                scalar = self.stylecode_to_multi_scalar_model(torch.flatten(style_code, start_dim=1), degree)
                scalar = scalar.view(-1, 1, 1)
                sym_stylecode = style_code + scalar * (torch.unsqueeze(pose_direction[0], 0).repeat(1, 1, 1))
                img, _ = self.g_ema_2d(sym_stylecode, truncation=1, truncation_latent=None, input_is_Wplus=True)
                img = torch.clamp(img, -1, 1)
                test_output_dir = os.path.join(self.options.test_output_dir, "multi_pose_scalar")
                ensure_dir_exist(test_output_dir)
                output_img_from_tensor(img, os.path.join(test_output_dir, 'test%04d_%d.png' % (i, item)))

    def test_stylegan2_to_3d_vanilla(self, amount=1):
        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            ######################################################
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            angles = np.linspace(-1, 1, 41).tolist()
            angles.extend(np.linspace(1, -1, 41).tolist())
            recon_rotate = []
            for ind, angle in enumerate(angles):
                coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                x = {'coeff': coeff,
                     'uvmap': uvmap,
                     'uv_displace_map': uv_displace_map,
                     'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                # out_img = out_img[:,:,::-1]
                recon_rotate.append(out_img)
            recon_rotate = np.array(recon_rotate)
            # print(recon_rotate.shape)
            recon_rotate = torch.tensor(recon_rotate)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN")
            ensure_dir_exist(test_output_dir)
            mimwrite(os.path.join(test_output_dir, 'test%02d.gif' % (num)), recon_rotate)

    def test_stylegan2_to_3d_expression(self, amount=1):
        def offset_expression_at_dimension(coeff, dim, val):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.Split_coeff(new_coeff)
            ex_coeff[:, dim] += val
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            sampled_e = np.random.randint(0, 64, 36)
            sampled_v = np.random.uniform(0.0, 50.0, 36)
            sampled_ev = np.vstack((sampled_e, sampled_v))
            for sample_vector in sampled_ev.T:
                e, v = sample_vector
                e = int(e)
                new_coeff = offset_expression_at_dimension(coeff, e, v)
                uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                angles = np.linspace(-1, 1, 41).tolist()
                angles.extend(np.linspace(1, -1, 41).tolist())
                recon_rotate = []
                for ind, angle in enumerate(angles):
                    new_coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                    x = {'coeff': new_coeff,
                            'uvmap': uvmap,
                            'uv_displace_map': uv_displace_map,
                            'uv_coord_2': self.uv_coord_2}
                    rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                    rendered_img = rendered_img.detach().cpu().numpy()[0]
                    a = np.clip(rendered_img[:, :, :3], 0, 1)
                    out_img = (a * 255).astype(np.uint8)
                    # out_img = out_img[:,:,::-1]
                    recon_rotate.append(out_img)
                recon_rotate = np.array(recon_rotate)
                # print(recon_rotate.shape)
                recon_rotate = torch.tensor(recon_rotate)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression")
                ensure_dir_exist(test_output_dir)
                mimwrite(os.path.join(test_output_dir, 'expression-%02d-%02d-%02d.gif' % (num, e, v)), recon_rotate)

    def test_stylegan2_to_3d_lighting(self, amount=1):
        def offset_lighting_at_dimension(coeff, dim, val):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            gamma[:, dim] += val
            gamma[:, dim + 9] += val
            gamma[:, dim + 18] += val
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            if num != 96:
                continue
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            angles = np.linspace(-1, 1, 41).tolist()
            angles.extend(np.linspace(1, -1, 41).tolist())
            recon_rotate = []
            x = {'coeff': coeff,
                    'uvmap': uvmap,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            a = np.clip(rendered_img[:, :, :3], 0, 1)
            out_img = (a * 255).astype(np.uint8)
            # out_img = out_img[:,:,::-1]
            # recon_rotate.append(out_img)
            # recon_rotate = np.array(recon_rotate)
            # print(recon_rotate.shape)
            # recon_rotate = torch.tensor(recon_rotate)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'lighting-origin.png'), out_img[..., ::-1])
            # mimwrite(os.path.join(test_output_dir, 'lighting-origin.gif'), recon_rotate)
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            sampled_l = np.random.randint(0, 9, 36)
            sampled_v = np.random.uniform(-0.5, 1.3, 36)
            sampled_lv = np.vstack((sampled_l, sampled_v))
            for sample_vector in sampled_lv.T:
                l, v = sample_vector
                l = int(l)
                new_coeff = offset_lighting_at_dimension(coeff, l, v)
                uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                angles = np.linspace(-1, 1, 41).tolist()
                angles.extend(np.linspace(1, -1, 41).tolist())
                recon_rotate = []
                for ind, angle in enumerate(angles):
                    new_coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                    x = {'coeff': new_coeff,
                            'uvmap': uvmap,
                            'uv_displace_map': uv_displace_map,
                            'uv_coord_2': self.uv_coord_2}
                    rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                    rendered_img = rendered_img.detach().cpu().numpy()[0]
                    a = np.clip(rendered_img[:, :, :3], 0, 1)
                    out_img = (a * 255).astype(np.uint8)
                    # out_img = out_img[:,:,::-1]
                    recon_rotate.append(out_img)
                recon_rotate = np.array(recon_rotate)
                # print(recon_rotate.shape)
                recon_rotate = torch.tensor(recon_rotate)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting")
                ensure_dir_exist(test_output_dir)
                mimwrite(os.path.join(test_output_dir, 'lighting-%02d-%02d-%0.2f.gif' % (num, l, v)), recon_rotate)

    def test_stylegan2_to_3d_lighting_variation(self, amount=1):
        def offset_lighting_at_dimension(coeff, dim, val):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            gamma[:, dim] += val
            gamma[:, dim + 9] += val
            gamma[:, dim + 18] += val
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))

            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting_variation")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            if num != 15:
                continue
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            x = {'coeff': coeff,
                    'uvmap': uvmap,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            a = np.clip(rendered_img[:, :, :3], 0, 1)
            out_img = (a * 255).astype(np.uint8)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting_variation")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'lighting-%02d-origin.png' % (num)), out_img[..., ::-1])

            # 0~63
            while True:
                print("Current lighting coefficients: ", coeff[:, 227:254].detach().cpu().numpy())
                print("enter lighting and value")
                inp = input()
                e, v = inp.split(' ', 1)
                e, v = int(e), float(v)
                new_coeff = offset_lighting_at_dimension(coeff, e, v)
                uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                angles = np.linspace(-1, 1, 41).tolist()
                angles.extend(np.linspace(1, -1, 41).tolist())
                recon_rotate = []
                x = {'coeff': new_coeff,
                        'uvmap': uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                # out_img = out_img[:,:,::-1]
                # recon_rotate.append(out_img)
                # recon_rotate = np.array(recon_rotate)
                # # print(recon_rotate.shape)
                # recon_rotate = torch.tensor(recon_rotate)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting_variation")
                ensure_dir_exist(test_output_dir)
                cv2.imwrite(os.path.join(test_output_dir, 'lighting-%02d-%02d-%.3f.png' % (num, e, v)), out_img[..., ::-1])
                # mimwrite(os.path.join(test_output_dir, 'expression-%02d-%02d-%.3f.gif' % (num, e, v)), recon_rotate)

    def test_stylegan2_to_3d_lighting_variation_animate(self, amount=1):
        def offset_lighting_at_dimension(coeff, dim, val):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            gamma[:, dim] += val
            gamma[:, dim + 9] += val
            gamma[:, dim + 18] += val
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))

            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting_variation")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            if num != 10:
                continue
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            angles = np.linspace(-1, 1, 41).tolist()
            angles.extend(np.linspace(1, -1, 41).tolist())
            recon_rotate = []
            for ind, angle in enumerate(angles):
                coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                x = {'coeff': coeff,
                        'uvmap': uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                # out_img = out_img[:,:,::-1]
                recon_rotate.append(out_img)
            recon_rotate = np.array(recon_rotate)
            # print(recon_rotate.shape)
            recon_rotate = torch.tensor(recon_rotate)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting_variation")
            ensure_dir_exist(test_output_dir)
            mimwrite(os.path.join(test_output_dir, 'lighting-%02d-origin.gif' % (num)), recon_rotate)


            # 0~63
            while True:
                print("Current lighting coefficients: ", coeff[:, 227:254].detach().cpu().numpy())
                print("enter lighting and value")
                inp = input()
                e, v = inp.split()
                e, v = int(e), float(v)
                new_coeff = offset_lighting_at_dimension(coeff, e, v)
                uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                angles = np.linspace(-1, 1, 41).tolist()
                angles.extend(np.linspace(1, -1, 41).tolist())
                recon_rotate = []
                for ind, angle in enumerate(angles):
                    new_coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                    x = {'coeff': new_coeff,
                            'uvmap': uvmap,
                            'uv_displace_map': uv_displace_map,
                            'uv_coord_2': self.uv_coord_2}
                    rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                    rendered_img = rendered_img.detach().cpu().numpy()[0]
                    a = np.clip(rendered_img[:, :, :3], 0, 1)
                    out_img = (a * 255).astype(np.uint8)
                    # out_img = out_img[:,:,::-1]
                    recon_rotate.append(out_img)
                recon_rotate = np.array(recon_rotate)
                # print(recon_rotate.shape)
                recon_rotate = torch.tensor(recon_rotate)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_lighting_variation")
                ensure_dir_exist(test_output_dir)
                mimwrite(os.path.join(test_output_dir, 'lighting-%02d-%02d-%.3f.gif' % (num, e, v)), recon_rotate)



    def test_stylegan2_to_3d_expression_variation(self, amount=1):
        def offset_expression_at_dimension(coeff, dim, val):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            ex_coeff[:, dim] = val
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression_variation")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            if num != 106:
                continue

            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            x = {'coeff': coeff,
                    'uvmap': uvmap,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            a = np.clip(rendered_img[:, :, :3], 0, 1)
            out_img = (a * 255).astype(np.uint8)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression_variation")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'expression-%02d-origin.png' % (num)), out_img[..., ::-1])

            # 0~63
            while True:
                print("Current expression coefficients: ", coeff[:, 80:144].detach().cpu().numpy())
                print("enter expression and value")
                inp = input()
                e, v = inp.split(' ', 1)
                e, v = int(e), float(v)
                coeff = offset_expression_at_dimension(coeff, e, v)
                uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                x = {'coeff': coeff,
                        'uvmap': uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression_variation")
                ensure_dir_exist(test_output_dir)
                cv2.imwrite(os.path.join(test_output_dir, 'expression-%02d-%02d-%.3f.png' % (num, e, v)), out_img[..., ::-1])

    def test_stylegan2_to_3d_expression_variation_animate(self, amount=1):
        def offset_expression_at_dimension(coeff, dim, val):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            ex_coeff[:, dim] = val
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression_variation")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            if num != 96:
                continue

            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            angles = np.linspace(-1, 1, 41).tolist()
            angles.extend(np.linspace(1, -1, 41).tolist())
            recon_rotate = []
            for ind, angle in enumerate(angles):
                coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                x = {'coeff': coeff,
                        'uvmap': uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                # out_img = out_img[:,:,::-1]
                recon_rotate.append(out_img)
            recon_rotate = np.array(recon_rotate)
            # print(recon_rotate.shape)
            recon_rotate = torch.tensor(recon_rotate)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression_variation")
            ensure_dir_exist(test_output_dir)
            mimwrite(os.path.join(test_output_dir, 'expression-%02d-origin.gif' % (num)), recon_rotate)

            # 0~63
            while True:
                print("Current expression coefficients: ", coeff[:, 80:144].detach().cpu().numpy())
                print("enter expression and value")
                inp = input()
                e, v = inp.split()
                e, v = int(e), float(v)
                coeff = offset_expression_at_dimension(coeff, e, v)
                uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                angles = np.linspace(-1, 1, 41).tolist()
                angles.extend(np.linspace(1, -1, 41).tolist())
                recon_rotate = []
                for ind, angle in enumerate(angles):
                    coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
                    x = {'coeff': coeff,
                            'uvmap': uvmap,
                            'uv_displace_map': uv_displace_map,
                            'uv_coord_2': self.uv_coord_2}
                    rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                    rendered_img = rendered_img.detach().cpu().numpy()[0]
                    a = np.clip(rendered_img[:, :, :3], 0, 1)
                    out_img = (a * 255).astype(np.uint8)
                    # out_img = out_img[:,:,::-1]
                    recon_rotate.append(out_img)
                recon_rotate = np.array(recon_rotate)
                # print(recon_rotate.shape)
                recon_rotate = torch.tensor(recon_rotate)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_expression_variation")
                ensure_dir_exist(test_output_dir)
                mimwrite(os.path.join(test_output_dir, 'expression-%02d-%02d-%0.3f.gif' % (num, e, v)), recon_rotate)


    def test_stylegan2_to_3d_interpolation(self, amount=1):
        def refill_lighting(coeff):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            gamma[:, :] = 0.045
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(2, 512).to(self.device))
            interpolated_inputs = [inputs[0][None, ...],
            (inputs[0] * 0.8 + inputs[1] * 0.2)[None, ...],
            (inputs[0] * 0.6 + inputs[1] * 0.4)[None, ...],
            (inputs[0] * 0.4 + inputs[1] * 0.6)[None, ...],
            (inputs[0] * 0.2 + inputs[1] * 0.8)[None, ...],
            inputs[1][None, ...]]
            # Output original stylecode output image
            for i, input_stylecode in enumerate(interpolated_inputs):
                image, _ = self.g_ema_2d(input_stylecode, input_is_Wplus=True)
                image = (image + 1) / 2
                image = image.permute(0, 2, 3, 1) * 255
                image = image.detach().cpu().numpy()[0]
                image = image[:, :, ::-1]
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_interpolation")
                ensure_dir_exist(test_output_dir)
                cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori-%d.png' % (num, i)), image)
                ######################################################
                coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(input_stylecode, start_dim=1))
                fake_coeff = refill_lighting(coeff)

                # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
                uvmap, uv_displace_map = self.g_ema(input_stylecode, input_is_Wplus=True, return_uv=True)
                uvmap = torch.tanh(uvmap)
                uvmap = (uvmap + 1) / 2
                uvmap = uvmap.permute(0, 2, 3, 1)
                x = {'coeff': coeff,
                        'uvmap': uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_interpolation")
                ensure_dir_exist(test_output_dir)
                cv2.imwrite(os.path.join(test_output_dir, 'rendered-%02d-ori-%d.png' % (num, i)), out_img[..., ::-1])


                fake_uvmap = torch.ones((1, 256, 256, 3)).to(self.device).detach()
                x = {'coeff': fake_coeff,
                        'uvmap': fake_uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                a = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (a * 255).astype(np.uint8)
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_interpolation")
                ensure_dir_exist(test_output_dir)
                cv2.imwrite(os.path.join(test_output_dir, 'geometry-%02d-ori-%d.png' % (num, i)), out_img[..., ::-1])

    def test_stylegan2_to_gradient_mask(self, amount=1):
        def get_gradmask(sym_coeff):
            grad_uvmap = Variable(torch.ones(1, 3, 256, 256) * 100, requires_grad=False).to(
                self.device)
            grad_uvmap = grad_uvmap.permute(0, 2, 3, 1)
            grad_uv_displace_map = Variable(torch.ones(1, 3, 256, 256) * 0, requires_grad=True).to(
                self.device)
            x = {'coeff': sym_coeff.detach(),
                'uvmap': grad_uvmap,
                'uv_displace_map': grad_uv_displace_map,
                'uv_coord_2': self.uv_coord_2}
            grad_rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            grad_uv_displace_map.retain_grad()
            (torch.sum(grad_rendered_img[:, :, :, :3] * 255)).backward()
            gradmask = torch.sum(torch.abs(grad_uv_displace_map.grad), dim=1)  # (b, 256, 256)
            gradmask = (gradmask != 0) * 1.
            # u, v = np.linspace(0, 1, 256), np.linspace(0, 1, 256)
            # u, v = np.meshgrid(u, v)
            # demo_grad_map = np.stack((u, v, np.zeros_like(u)), axis=0)[None, ...]
            demo_grad_map = cv2.imread('grid.png')
            # gradmask = torch.unsqueeze(gradmask, 1)
            # gradmask = make_grid(gradmask, nrow=4, normalize=False,scale_each=True)
            # gradmask = gradmask.permute(1,2,0).detach().cpu().numpy()
            # cv2.imwrite('./testing/image%04d.png'%(idx),gradmask*255)
            return gradmask.permute(1, 2, 0).repeat(1, 1, 3) * (torch.Tensor(demo_grad_map).to(self.device) / 255.0)

        def refill_lighting(coeff):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            gamma[:, :] = 0.045
            return new_coeff

        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_gradient")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            demo_gradmask = get_gradmask(coeff)[None, ...]
            cv2.imwrite(os.path.join(test_output_dir, 'grad-%02d-ori.png' % (num)), demo_gradmask.detach()[0].cpu().numpy()[..., ::-1]*255.0)

            fake_coeff = refill_lighting(coeff)

            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            x = {'coeff': coeff,
                    'uvmap': demo_gradmask,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2,
                    'need_illu': False}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            a = np.clip(rendered_img[:, :, :3], 0, 1)
            out_img = (a * 255).astype(np.uint8)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_gradient")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'rendered-%02d-ori.png' % (num)), out_img[..., ::-1])


            # fake_uvmap = torch.ones((1, 256, 256, 3)).to(self.device).detach()
            # x = {'coeff': fake_coeff,
            #         'uvmap': fake_uvmap,
            #         'uv_displace_map': uv_displace_map,
            #         'uv_coord_2': self.uv_coord_2}
            # rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            # rendered_img = rendered_img.detach().cpu().numpy()[0]
            # a = np.clip(rendered_img[:, :, :3], 0, 1)
            # out_img = (a * 255).astype(np.uint8)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_interpolation")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'geometry-%02d-ori.png' % (num)), out_img[..., ::-1])
    def test_fig3_both(self, amount=1):
        def refill_lighting(coeff):
            new_coeff = coeff.clone().detach()
            id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = FaceRenderer.split_3dmm_coeff(new_coeff)
            gamma[:, :] = 0.045
            return new_coeff
        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            if num != 55:
                continue
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_origin")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-ori.png' % (num)), image)
            # Symmetric StyleGAN2 view
            scalar = self.stylecode_to_scalar_model(torch.flatten(inputs, start_dim=1)).view(-1, 1, 1)
            pose_direction = torch.load(os.path.join(self.options.dataset_dir, 'pose_direction-new.pkl')).view(14, 512).type(torch.FloatTensor).to(self.device)
            symmetric_style_code = inputs + scalar * pose_direction
            sym_image, _ = self.g_ema_2d(symmetric_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
            sym_image = torch.clamp(sym_image, -1, 1)
            sym_image = ((sym_image + 1) / 2).detach()
            sym_image = sym_image.permute(0, 2, 3, 1) * 255
            sym_image = sym_image.detach().cpu().numpy()[0]
            sym_image = sym_image[:, :, ::-1]
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-sym.png' % (num)), sym_image)
            ######################################################
            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))
            sym_coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(symmetric_style_code, start_dim=1))

            fake_coeff = refill_lighting(coeff)
            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            # ori
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_diffuse")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-diffuse.png' % (num)), (uvmap.detach().cpu().numpy()[0][:, :, ::-1]*255.0))

            sym_uvmap, sym_uv_displace_map = self.g_ema(symmetric_style_code, input_is_Wplus=True, return_uv=True)
            sym_uvmap = torch.tanh(sym_uvmap)
            sym_uvmap = (sym_uvmap + 1) / 2
            sym_uvmap = sym_uvmap.permute(0, 2, 3, 1)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_diffuse")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-diffuse-sym.png' % (num)), (sym_uvmap.detach().cpu().numpy()[0][:, :, ::-1]*255.0))

            sym_gradmask = self.get_gradmask(sym_coeff.detach()).detach()
            gradmask = self.get_gradmask(coeff.detach()).detach()  # (b, 256, 256) 0~1
            sym_gradleftmask = (sym_gradmask - (sym_gradmask > 0) * (gradmask > 0) * 1.)
            # simplify to this.
            # sym_gradleftmask = (self.uvmask > 0) * (~(sym_gradleftmask > 0)) * 0.5 + sym_gradleftmask
            sym_gradleftmask = (self.uvmask > 0) * (~((self.uvmask > 0) * (sym_gradleftmask > 0))) * 0.5 + sym_gradleftmask
            gradmask = torch.unsqueeze(gradmask, 3).repeat(1, 1, 1, 3)
            sym_gradleftmask = torch.unsqueeze(sym_gradleftmask, 3).repeat(1, 1, 1, 3)
            x = {'coeff': coeff,
                    'uvmap': gradmask,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2,
                    'need_illu': False}
            weighted_mask, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            weighted_mask = weighted_mask[:, :, :, 0][..., None].repeat(1, 1, 1, 3).detach().cpu().numpy()
            out_img = (np.clip(weighted_mask[0], 0, 1) * 255)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_weight")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-weight.png' % (num)), out_img)

            x = {'coeff': sym_coeff,
                    'uvmap': sym_gradleftmask,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2,
                    'need_illu': False}
            weighted_mask, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            weighted_mask = weighted_mask[:, :, :, 0][..., None].repeat(1, 1, 1, 3).detach().cpu().numpy()
            out_img = (np.clip(weighted_mask[0], 0, 1) * 255)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_weight")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-weight-sym.png' % (num)), out_img)


            angles = np.linspace(-1, 1, 41).tolist()
            angles.extend(np.linspace(1, -1, 41).tolist())
            # recon_rotate = []
            # fake_uvmap = torch.ones((1, 256, 256, 3)).to(self.device).detach()
            # fake_displace_map = torch.zeros((1, 3, 256, 256)).to(self.device).detach()
            # for ind, angle in enumerate(angles):
            #     fake_coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
            #     x = {'coeff': fake_coeff,
            #             'uvmap': fake_uvmap,
            #             'uv_displace_map': fake_displace_map,
            #             'uv_coord_2': self.uv_coord_2,
            #             'need_illu': False}
            #     rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            #     rendered_img = rendered_img.detach().cpu().numpy()[0]
            #     a = np.clip(rendered_img[:, :, :3], 0, 1)
            #     out_img = (a * 255).astype(np.uint8)
            #     # out_img = out_img[:,:,::-1]
            #     recon_rotate.append(out_img)
            # recon_rotate = np.array(recon_rotate)
            # # print(recon_rotate.shape)
            # recon_rotate = torch.tensor(recon_rotate)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_generated")
            # ensure_dir_exist(test_output_dir)
            # mimwrite(os.path.join(test_output_dir, 'thumbnail-%02d.gif' % (num)), recon_rotate)

            # recon_rotate = []
            # for ind, angle in enumerate(angles):
            #     fake_coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
            #     x = {'coeff': fake_coeff,
            #             'uvmap': fake_uvmap,
            #             'uv_displace_map': uv_displace_map,
            #             'uv_coord_2': self.uv_coord_2}
            #     rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            #     rendered_img = rendered_img.detach().cpu().numpy()[0]
            #     a = np.clip(rendered_img[:, :, :3], 0, 1)
            #     out_img = (a * 255).astype(np.uint8)
            #     # out_img = out_img[:,:,::-1]
            #     recon_rotate.append(out_img)
            # recon_rotate = np.array(recon_rotate)
            # # print(recon_rotate.shape)
            # recon_rotate = torch.tensor(recon_rotate)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_thumbnail")
            # ensure_dir_exist(test_output_dir)
            # mimwrite(os.path.join(test_output_dir, 'thumbnail-%02d-1.gif' % (num)), recon_rotate)

            # recon_rotate = []
            x = {'coeff': coeff,
                    'uvmap': uvmap,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2}
            rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            rendered_img = rendered_img.detach().cpu().numpy()[0]
            a = np.clip(rendered_img[:, :, :3], 0, 1)
            out_img = (a * 255).astype(np.uint8)
            # out_img = out_img[:,:,::-1]
            # recon_rotate.append(out_img)
            # recon_rotate = np.array(recon_rotate)
            # print(recon_rotate.shape)
            # recon_rotate = torch.tensor(recon_rotate)
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_result_both")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'fig3-%02d-2.png' % (num)), out_img[:, :, ::-1])
            cv2.imwrite(os.path.join(test_output_dir, 'fig3-%02d-2-diffuse.png' % (num)), uvmap.detach().cpu().numpy()[0][:, :, ::-1]*255.0)
            # mimwrite(os.path.join(test_output_dir, 'fig3-%02d-2.gif' % (num)), recon_rotate)

            # recon_rotate = []
            # for ind, angle in enumerate(angles):
            #     sym_coeff[0][self.options.coeff_yaw_angle_dim] = torch.tensor(angle).to(self.device)
            #     x = {'coeff': sym_coeff,
            #             'uvmap': sym_uvmap,
            #             'uv_displace_map': sym_uv_displace_map,
            #             'uv_coord_2': self.uv_coord_2}
            #     rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
            #     rendered_img = rendered_img.detach().cpu().numpy()[0]
            #     a = np.clip(rendered_img[:, :, :3], 0, 1)
            #     out_img = (a * 255).astype(np.uint8)
            #     # out_img = out_img[:,:,::-1]
            #     recon_rotate.append(out_img)
            # recon_rotate = np.array(recon_rotate)
            # # print(recon_rotate.shape)
            # recon_rotate = torch.tensor(recon_rotate)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_result")
            # ensure_dir_exist(test_output_dir)
            # mimwrite(os.path.join(test_output_dir, 'fig3-%02d-2-sym.gif' % (num)), recon_rotate)
    def test_fig3_paste(self, amount=1):
        for num in np.arange(amount).tolist():
            inputs = self.g_ema_2d.get_latent_Wplus(torch.randn(1, 512).to(self.device))
            # Output original stylecode output image
            image, _ = self.g_ema_2d(inputs, input_is_Wplus=True)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1) * 255
            image = image.detach().cpu().numpy()[0]
            image = image[:, :, ::-1]
            if num != 82:
                continue
            test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_paste")
            ensure_dir_exist(test_output_dir)
            cv2.imwrite(os.path.join(test_output_dir, 'fig3-%02d-1.png' % (num)), image)
            # Symmetric StyleGAN2 view
            scalar = self.stylecode_to_scalar_model(torch.flatten(inputs, start_dim=1)).view(-1, 1, 1)
            print(scalar)
            for angle in range(-100, 50):
                scalar = angle
                turned_style_code = inputs + scalar * self.pose_direction
                turned_image, _ = self.g_ema_2d(turned_style_code, input_is_Wplus=True)
                turned_image = (turned_image + 1) / 2
                turned_image = turned_image.permute(0, 2, 3, 1) * 255
                turned_image = turned_image.detach().cpu().numpy()[0]
                turned_image = turned_image[:, :, ::-1]

                cv2.imwrite(os.path.join(test_output_dir, 'fig3-%02d-1-%d.png' % (num, angle)), turned_image)

            coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(inputs, start_dim=1))

            # Expression parameter space, for detail, please see Split_coeff() in network/arch_3DMM.py
            # ori
            uvmap, uv_displace_map = self.g_ema(inputs, input_is_Wplus=True, return_uv=True)
            uvmap = torch.tanh(uvmap)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.permute(0, 2, 3, 1)
            # test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_fig3_diffuse")
            # ensure_dir_exist(test_output_dir)
            # cv2.imwrite(os.path.join(test_output_dir, 'test%02d-diffuse.png' % (num)), (uvmap.detach().cpu().numpy()[0][:, :, ::-1]*255.0))
            # for stylegan2, for 3DMM
            # -23
            angle_list = [-52]
            for angle in angle_list:
                turned_style_code = inputs + angle * self.pose_direction
                turned_image, _ = self.g_ema_2d(turned_style_code, input_is_Wplus=True)
                turned_image = (turned_image + 1) / 2
                turned_image = turned_image.permute(0, 2, 3, 1) * 255
                turned_image = turned_image.detach().cpu().numpy()[0]
                turned_image = turned_image[:, :, ::-1]

                turned_coeff = self.stylecode_to_3dmm_coeff_model(torch.flatten(turned_style_code, start_dim=1))
                # angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
                # translation = coeff[:, 254:]  # translation coeff of dim 3

                coeff[0, 224:227] = turned_coeff[0, 224:227]
                coeff[0, 254:] = turned_coeff[0, 254:]  # translation coeff of dim 3
                x = {'coeff': coeff,
                    'uvmap': uvmap,
                    'uv_displace_map': uv_displace_map,
                    'uv_coord_2': self.uv_coord_2}
                rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                rendered_img = rendered_img.detach().cpu().numpy()[0]
                output_rendered_img = np.clip(rendered_img[:, :, :3], 0, 1)
                out_img = (output_rendered_img * 255).astype(np.uint8)[:, :, ::-1]
                paste_img = turned_image.copy()
                paste_img[rendered_img[:, :, 3] > 0] = out_img[rendered_img[:, :, 3] > 0]
                test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_paste")
                ensure_dir_exist(test_output_dir)
                cv2.imwrite(os.path.join(test_output_dir, 'fig3-%02d-pasted-%d.png' % (num, angle)), paste_img)
                while True:
                    print("please input dimension to alter and angle..")
                    print('current dimension value: angle=', coeff[0, 224].detach().cpu(), coeff[0, 225].detach().cpu(), coeff[0, 226].detach().cpu(), 
                        " translation=", coeff[0, 254].detach().cpu(), coeff[0, 255].detach().cpu(), coeff[0, 256].detach().cpu())
                    dimension, input_angle = input().split()
                    dimension = int(dimension)
                    input_angle = float(input_angle)
                    coeff[0, dimension] = input_angle
                    x = {'coeff': coeff,
                        'uvmap': uvmap,
                        'uv_displace_map': uv_displace_map,
                        'uv_coord_2': self.uv_coord_2}
                    rendered_img, _, _, _, _, _ = self.pure_3dmm_uv_model(x)
                    rendered_img = rendered_img.detach().cpu().numpy()[0]
                    output_rendered_img = np.clip(rendered_img[:, :, :3], 0, 1)
                    out_img = (output_rendered_img * 255).astype(np.uint8)[:, :, ::-1]
                    paste_img = turned_image.copy()
                    paste_img[rendered_img[:, :, 3] > 0] = out_img[rendered_img[:, :, 3] > 0]
                    test_output_dir = os.path.join(self.options.test_output_dir, "DispGAN_paste")
                    ensure_dir_exist(test_output_dir)
                    cv2.imwrite(os.path.join(test_output_dir, 'fig3-%02d-pasted-%d-cur.png' % (num, angle)), paste_img)
