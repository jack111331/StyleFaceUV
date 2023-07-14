import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils.losses import l1loss
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from utils.utility import instantiate_from_config
import yaml

class StylecodeTo3DCoeffTrainer(pl.LightningModule):
    def __init__(self, 
                 style_to_3dmm_coeff_config,
                 ckpt_path=None):
        
        super().__init__()
        self.stylecode_to_3dmm_coeff_model = instantiate_from_config(style_to_3dmm_coeff_config)
        
        self.save_hyperparameters()

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])
    
    def configure_optimizers(self):
        # The self.learning_rate is configured after instantiating this model
        optimize_params = list(self.stylecode_to_3dmm_coeff_model.parameters())
        optimizer = torch.optim.Adam(optimize_params, lr=self.learning_rate)

        scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        style_code, coeff = train_batch
        output = self.stylecode_to_3dmm_coeff_model(torch.flatten(style_code, start_dim=1))
        loss = l1loss(output, coeff)
        self.log("train/l1_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        style_code, coeff = batch
        output = self.stylecode_to_3dmm_coeff_model(torch.flatten(style_code, start_dim=1))
        loss = l1loss(output, coeff)
        self.log("val/l1_loss", loss, logger=True)
        return loss

if __name__ == '__main__':
    with open('config/stylecode_to_3dcoeff_config.yml', 'r') as stream:
        config = yaml.safe_load(stream)

    trainer_config = config['trainer']
    model = instantiate_from_config(trainer_config)
    model.learning_rate = trainer_config['learning_rate']

    batch_size = trainer_config['batch_size']

    train_proportion = trainer_config['train_proportion']

    dataset_config = trainer_config['dataset_config']
    dataset = instantiate_from_config(dataset_config)

    train_dataset, val_dataset = dataset[:int(len(dataset) * train_proportion)], dataset[int(len(dataset) * train_proportion):]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    trainer = pl.Trainer(default_root_dir="ckpt/stylecode_to_3dmm_coeff", gpus=1, max_epochs=25)
    trainer.fit(model, train_dataloader, val_dataloader)
