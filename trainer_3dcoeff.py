import os, copy

import torch
from torch.optim.lr_scheduler import MultiStepLR
import scipy.io

from utils.options import Option
from utils.base_trainer import TrainValidBaseTrainer
from network.arch import Generator2D, StylecodeTo3DMMCoeffMLP
from network.arch_3DMM import Pure_3DMM
from utils.losses import l1loss
from utils.utility import output_img_from_tensor, output_3dmm_masked_img_from_tensor, train_valid_split, ensure_dir_exist
from dataloader.dataset import StyleCode3DMMParamsDataset
from utils.saver import BestFirstCheckpointSaver

debug_output = False


class Trainer(TrainValidBaseTrainer):
    def init_fn(self):
        ensure_dir_exist(self.options.ckpt_output_dir)

        self.g_ema = Generator2D(size=self.options.size, style_dim=self.options.latent, n_mlp=self.options.n_mlp).to(self.device)
        self.g_ema.load_state_dict(torch.load(self.options.generator_2d_ckpt)['g_ema'], strict=False)

        self.datasets, self.data_loaders = train_valid_split(StyleCode3DMMParamsDataset(self.options.dataset_dir), 0.9,
                                                             self.options.batch_size)

        self.stylecode_to_3dmm_coeff_model = StylecodeTo3DMMCoeffMLP().to(self.device)

        facemodel = scipy.io.loadmat(self.options.FACE_MODEL_PATH)

        self.pure_3dmm_model = Pure_3DMM(facemodel).to(self.device)

        self.g_ema.eval()
        self.pure_3dmm_model.eval()

        self.optimizer = torch.optim.Adam(self.stylecode_to_3dmm_coeff_model.parameters(), lr=self.options.l_rate)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[5, 10, 15], gamma=0.1)

        self.valid_data = torch.unsqueeze(torch.load(self.options.debug_stylecode)[0], 0).to(self.device)

        self.saver = BestFirst3DCoeffCheckpointSaver(self.options.ckpt_output_dir)
        if self.saver.checkpoint != None:
            additional_param = self.saver.load_checkpoint(models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model}, optimizers={"Stylecode to 3DMM Coeff Optimizer": self.optimizer})

            self.epoch = additional_param['epoch']
            self.batch_idx = additional_param['batch_idx']
            self.best_loss = additional_param['best_loss']
            self.options.l_rate = additional_param['l_rate']
            self.options.num_epochs = additional_param['num_epochs']
            self.options.batch_size = additional_param['batch_size']
        else:
            self.best_loss = None

    def train_step(self, idx, input_batch, stage):

        style_code = input_batch[0].to(self.device)
        coeff = input_batch[1].to(self.device)
        with torch.set_grad_enabled(stage == 'train'):
            output = self.stylecode_to_3dmm_coeff_model(torch.flatten(style_code, start_dim=1))
            self.optimizer.zero_grad()
            loss = l1loss(output, coeff)
            if stage == 'train':
                loss.backward()
                self.optimizer.step()
        return loss.item()

    def pre_train_step(self):
        self.stylecode_to_3dmm_coeff_model.train()

    def pre_valid_step(self):
        self.stylecode_to_3dmm_coeff_model.eval()

    def debug_post_step(self, epoch, epoch_loss):
        if debug_output == True:
            img_tensor, _ = self.g_ema(self.valid_data, truncation=1, truncation_latent=None, input_is_Wplus=True)
            debug_output_dir = os.path.join(self.options.debug_output_dir, "3dcoeff")
            ensure_dir_exist(debug_output_dir)

            img = output_img_from_tensor(img_tensor,
                                         os.path.join(debug_output_dir, 'stylegan_image.png'))

            output = self.stylecode_to_3dmm_coeff_model(torch.flatten(self.valid_data, start_dim=1))

            x = {'coeff': output}
            rendered_img_tensor, _, _, _, _, _ = self.pure_3dmm_model(x)
            debug_output_dir = os.path.join(self.options.debug_output_dir, "3dcoeff")
            ensure_dir_exist(debug_output_dir)
            output_3dmm_masked_img_from_tensor(rendered_img_tensor, os.path.join(debug_output_dir,
                                                                                 '3dmm_image%02d.png' % (epoch)), img)
    def post_train_step(self, epoch, epoch_loss):
        super(Trainer, self).post_train_step(epoch, epoch_loss)
        self.debug_post_step(epoch, epoch_loss)

    def post_valid_step(self, epoch, epoch_loss):
        super(Trainer, self).post_valid_step(epoch, epoch_loss)
        self.debug_post_step(epoch, epoch_loss)
        if epoch == 0:
            self.best_loss = epoch_loss

            self.saver.save_checkpoint(models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model}, optimizers={"Stylecode to 3DMM Coeff Optimizer": self.optimizer}, trainer=self)

            # best_model_wts = copy.deepcopy(self.stylecode_to_3dmm_coeff_model.state_dict())
            # torch.save(best_model_wts, os.path.join(self.options.ckpt_output_dir, "param%02d.pkl" % (epoch)))
        else:
            super(BestFirstCheckpointSaver, self.saver).save_checkpoint(models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model},
                                       optimizers={"Stylecode to 3DMM Coeff Optimizer": self.optimizer},
                                       trainer=self)
            if epoch_loss < self.best_loss:
                print("Got New Best !")
                self.best_loss = epoch_loss

                self.saver.save_checkpoint(models={"Stylecode to 3DMM Coeff": self.stylecode_to_3dmm_coeff_model},
                                           optimizers={"Stylecode to 3DMM Coeff Optimizer": self.optimizer},
                                           trainer=self)

                # best_model_wts = copy.deepcopy(self.stylecode_to_3dmm_coeff_model.state_dict())
                # torch.save(best_model_wts, os.path.join(self.options.ckpt_output_dir, "param%02d.pkl" % (epoch)))

class BestFirst3DCoeffCheckpointSaver(BestFirstCheckpointSaver):
    # inheritance
    def get_model_related_attribute(self, trainer):
        return {'l_rate': trainer.options.l_rate, 'num_epochs': trainer.options.num_epochs,
                'batch_size': trainer.options.batch_size, 'best_loss': trainer.best_loss}

    # inheritance
    def get_model_related_attribute_from_checkpoint(self, checkpoint):
        return {'l_rate': checkpoint['l_rate'], 'num_epochs': checkpoint['num_epochs'],
                'batch_size': checkpoint['batch_size'], 'best_loss': checkpoint['best_loss']}


if __name__ == '__main__':
    option = Option("Style Code To 3DMM coeff MLP")
    args = option.parse_args()
    trainer = Trainer(args)
    trainer.train()
