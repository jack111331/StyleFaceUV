import os, copy

import torch
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from utils.options import Option
from utils.base_trainer import TrainValidBaseTrainer
from network.arch import Generator2D
from network.arch import StylecodeToMultiPoseDirectionScalarMLP, StylecodeTo3DMMCoeffMLP
from utils.losses import l1loss
from utils.utility import output_grid_img_from_tensor, train_valid_split, ensure_dir_exist
from dataloader.dataset import StyleCode3DMMParamsDataset
from utils.saver import BestFirstCheckpointSaver
from trainer_3dcoeff import BestFirst3DCoeffCheckpointSaver

debug_output = False

class Trainer(TrainValidBaseTrainer):
    def init_fn(self):
        ensure_dir_exist(self.options.ckpt_output_dir)

        self.g_ema = Generator2D(size=self.options.size, style_dim=self.options.latent, n_mlp=self.options.n_mlp).to(self.device)
        self.g_ema.load_state_dict(torch.load(self.options.generator_2d_ckpt)['g_ema'], strict=False)

        self.stylecode_to_scalar_model = StylecodeToMultiPoseDirectionScalarMLP().to(self.device)

        self.datasets, self.data_loaders = train_valid_split(StyleCode3DMMParamsDataset(self.options.dataset_dir), 0.9, self.options.batch_size)

        self.stylecode_to_3dmm_coeff_model = StylecodeTo3DMMCoeffMLP().to(self.device)
        stylecode_to_3dmm_saver = BestFirst3DCoeffCheckpointSaver(self.options.stylecode_3dmm_coeff_mlp_ckpt)
        if stylecode_to_3dmm_saver.checkpoint != None:
            stylecode_to_3dmm_saver.load_checkpoint(
                models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model},
                optimizers={})
        self.optimizer = torch.optim.Adam(self.stylecode_to_scalar_model.parameters(),lr=self.options.l_rate)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[5,10,15], gamma=0.1)

        self.pose_normal_direction = torch.load(self.options.pose_normal_direction).view(1, 14, 512).type(
            torch.FloatTensor)
        self.pose_normal_direction = self.pose_normal_direction.repeat(self.options.batch_size, 1, 1).to(self.device)

        self.valid_data = torch.load(self.options.debug_stylecode)[16:32].to(self.device)

        self.saver = BestFirstMultiPoseScalarCheckpointSaver(self.options.ckpt_output_dir)
        if self.saver.checkpoint != None:
            additional_param = self.saver.load_checkpoint(models={"Stylecode to Multi Pose Scalar": self.stylecode_to_scalar_model}, optimizers={"Stylecode to Multi Pose Scalar Optimizer": self.optimizer})

            self.epoch = additional_param['epoch']
            self.batch_idx = additional_param['batch_idx']
            self.best_loss = additional_param['best_loss']
            self.options.l_rate = additional_param['l_rate']
            self.options.num_epochs = additional_param['num_epochs']
            self.options.batch_size = additional_param['batch_size']
        else:
            self.best_loss = None


        if debug_output == True:
            img_tensor, _ = self.g_ema(self.valid_data, truncation=1, truncation_latent=None, input_is_Wplus=True)

            debug_output_dir = os.path.join(self.options.debug_output_dir, "multi_pose_scalar")
            ensure_dir_exist(debug_output_dir)
            output_grid_img_from_tensor(img_tensor, os.path.join(debug_output_dir, 'ori_image.png'))


    def train_step(self, idx, input_batch, stage):
        self.g_ema.eval()
        self.stylecode_to_3dmm_coeff_model.eval()

        stylecode = input_batch[0].to(self.device)
        # sample_radian = torch.deg2rad(torch.FloatTensor(self.options.batch_size).uniform_(-30,
        #                                                                           30))  # it's uniform sampled between [-30 degrees, 30 degrees]
        sample_radian = np.pi * torch.FloatTensor(self.options.batch_size).uniform_(-30,
                                                                                  30) / 180.0  # it's uniform sampled between [-30 degrees, 30 degrees]
        sample_radian = sample_radian.to(self.device)
        with torch.set_grad_enabled(stage == 'train'):
            scalar = self.stylecode_to_scalar_model(torch.flatten(stylecode, start_dim=1),
                                                    torch.unsqueeze(sample_radian, 1))
            scalar = scalar.view(-1, 1, 1)
            sym_stylecode = stylecode + scalar * self.pose_normal_direction
            sym_3dmm_coeffs = self.stylecode_to_3dmm_coeff_model(torch.flatten(sym_stylecode, start_dim=1))
            self.optimizer.zero_grad()
            loss = l1loss(sym_3dmm_coeffs[:, 225], sample_radian)
            if stage == 'train':
                loss.backward()
                self.optimizer.step()
        return loss.item()

    def pre_train_step(self):
        self.stylecode_to_scalar_model.train()

    def pre_valid_step(self):
        self.stylecode_to_scalar_model.eval()

    def post_valid_step(self, epoch, epoch_loss):
        super(Trainer, self).post_valid_step(epoch, epoch_loss)
        if epoch == 0:
            self.best_loss = epoch_loss

            self.saver.save_checkpoint(models={"Stylecode to Multi Pose Scalar": self.stylecode_to_scalar_model}, optimizers={"Stylecode to Multi Pose Scalar Optimizer": self.optimizer}, trainer=self)

            # best_model_wts = copy.deepcopy(self.stylecode_to_scalar_model.state_dict())
            # torch.save(best_model_wts, os.path.join(self.options.ckpt_output_dir, "param.pkl"))
        else:
            super(BestFirstCheckpointSaver, self.saver).save_checkpoint(models={"Stylecode to Multi Pose Scalar": self.stylecode_to_scalar_model}, optimizers={"Stylecode to Multi Pose Scalar Optimizer": self.optimizer},
                                       trainer=self)

            if epoch_loss < self.best_loss:
                print("Got New Best !")
                self.best_loss = epoch_loss

                self.saver.save_checkpoint(models={"Stylecode to Multi Pose Scalar": self.stylecode_to_scalar_model}, optimizers={"Stylecode to Multi Pose Scalar Optimizer": self.optimizer},
                                           trainer=self)

                # best_model_wts = copy.deepcopy(self.stylecode_to_scalar_model.state_dict())
                # torch.save(best_model_wts, os.path.join(self.options.ckpt_output_dir, "param.pkl"))
                scalar = self.stylecode_to_scalar_model(torch.flatten(self.valid_data, start_dim=1), torch.zeros(16, 1).to(self.device))
                scalar = scalar.view(-1, 1, 1)
                if debug_output == True:
                    img_tensor, _ = self.g_ema(self.valid_data + scalar * (torch.unsqueeze(self.pose_normal_direction[0], 0).repeat(16, 1, 1)),
                                          truncation=1, truncation_latent=None, input_is_Wplus=True)

                    debug_output_dir = os.path.join(self.options.debug_output_dir, "multi_pose_scalar")
                    ensure_dir_exist(debug_output_dir)
                    output_grid_img_from_tensor(img_tensor, os.path.join(debug_output_dir, 'image%02d.png' % (epoch)))

class BestFirstMultiPoseScalarCheckpointSaver(BestFirstCheckpointSaver):
    # inheritance
    def get_model_related_attribute(self, trainer):
        return {'l_rate': trainer.options.l_rate, 'num_epochs': trainer.options.num_epochs,
                'batch_size': trainer.options.batch_size, 'best_loss': trainer.best_loss}

    # inheritance
    def get_model_related_attribute_from_checkpoint(self, checkpoint):
        return {'l_rate': checkpoint['l_rate'], 'num_epochs': checkpoint['num_epochs'],
                'batch_size': checkpoint['batch_size'], 'best_loss': checkpoint['best_loss']}



if __name__ == '__main__':
    option = Option("Style Code To Multi Pose Direction MLP")
    args = option.parse_args()
    trainer = Trainer(args)
    trainer.train()