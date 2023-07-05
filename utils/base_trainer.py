from __future__ import division
import torch
from tqdm import tqdm


tqdm.monitor_interval = 0


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_loaders = None
        self.datasets = None

        self.scheduler = None
        # last checkpoint epoch
        self.epoch = -1
        self.batch_idx = 0

        self.saver = None

        # override this function to define your model, optimizers etc.
        self.init_fn()

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch + 1, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch + 1):
            self.epoch = epoch
            # Create new DataLoader every epoch and (possibly) resume
            # from an arbitrary step inside an epoch

            # training stage, validation stage
            for stage in ['train']:
                self.pre_train_step()

                # Iterate over all batches in an epoch
                running_loss = 0.0
                for idx, batch in tqdm(enumerate(self.data_loaders), total=len(self.data_loaders)):
                    self.batch_idx = idx
                    batch = [b.to(self.device) for b in batch]
                    running_loss += self.train_step(idx, batch, stage)

                epoch_loss = running_loss / len(self.data_loaders)
                self.post_train_step(epoch, epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()
        return

    # The following methods (with the possible exception of test)
    # have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def pre_train_step(self):
        pass

    def post_train_step(self, epoch, epoch_loss):
        print("Phase: %s, Epoch: %02d, Loss: %f" % ("train", epoch, epoch_loss))

    def train_step(self, idx, input_batch, stage):
        raise NotImplementedError('You need to provide a _train_step method')

    def test(self):
        pass

class TrainValidBaseTrainer(BaseTrainer):
    def __init__(self, options):
        # [train_data_loader, valid_data_loader]
        self.data_loaders = {"train": None, "val": None}
        self.datasets = {"train": None, "val": None}

        super(TrainValidBaseTrainer, self).__init__(options)

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch + 1, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch + 1):
            self.epoch = epoch
            # Create new DataLoader every epoch and (possibly) resume
            # from an arbitrary step inside an epoch

            # training stage, validation stage
            for stage in ['train', 'val']:
                if stage == 'train':
                    self.pre_train_step()
                else:
                    self.pre_valid_step()

                # Iterate over all batches in an epoch
                running_loss = 0.0
                for idx, batch in tqdm(enumerate(self.data_loaders[stage]), total=len(self.data_loaders[stage])):
                    self.batch_idx = idx
                    batch = [b.to(self.device) for b in batch]
                    running_loss += self.train_step(idx, batch, stage)

                epoch_loss = running_loss / len(self.data_loaders[stage])
                if stage == 'train':
                    self.post_train_step(epoch, epoch_loss)
                else:
                    self.post_valid_step(epoch, epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()
        return

    def pre_valid_step(self):
        pass

    def post_valid_step(self, epoch, epoch_loss):
        print("Phase: %s, Epoch: %02d, Loss: %f" % ("valid", epoch, epoch_loss))

class GANBaseTrainer(BaseTrainer):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch + 1, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch + 1):
            self.epoch = epoch
            # Create new DataLoader every epoch and (possibly) resume
            # from an arbitrary step inside an epoch

            # training stage, validation stage
            for stage in ['train']:
                self.pre_train_step()

                # Iterate over all batches in an epoch
                # [running_loss, running_loss_g, running_loss_d]
                running_losses = [0.0, 0.0, 0.0]
                for idx, batch in tqdm(enumerate(self.data_loaders), total=len(self.data_loaders)):
                    self.batch_idx = idx
                    batch = [b.to(self.device) for b in batch]
                    train_losses = self.train_step(idx, batch, stage)
                    running_losses = [loss + train_losses[idx] for idx, loss in enumerate(running_losses)]

                epoch_loss = []
                epoch_loss.append(running_losses[0] / len(self.data_loaders))
                epoch_loss.append(running_losses[1] / len(self.data_loaders) * self.options.n_critic)
                epoch_loss.append(running_losses[2] / len(self.data_loaders) * self.options.n_critic_d)
                self.post_train_step(epoch, epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()
        return

    def post_train_step(self, epoch, epoch_loss):
        print("Epoch: %02d, Loss: %f, D Loss: %f, G_loss: %f" % (epoch, epoch_loss[0], epoch_loss[1], epoch_loss[2]))

