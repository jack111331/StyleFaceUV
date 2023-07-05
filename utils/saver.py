import os
import torch
import logging
import datetime

class ConstantCheckpointSaver(object):
    """Class that handles saving and loading checkpoints during training."""
    def __init__(self, save_dir, save_steps=1000):
        self.save_dir = os.path.abspath(save_dir)
        self.save_steps = save_steps
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.checkpoint = None
        self.get_checkpoint()
        return

    def exists_checkpoint(self, checkpoint_file=None):
        """Check if a checkpoint exists in the current directory."""
        if checkpoint_file is None:
            return False if self.checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)

    # inheritance
    def get_model_related_attribute_from_checkpoint(self, checkpoint):
        return {}

    # inheritance
    def get_model_related_attribute(self, trainer):
        return {}

    @staticmethod
    def get_model_unrelated_attribute_from_checkpoint(checkpoint):
        return {'epoch': checkpoint['epoch'], 'batch_idx': checkpoint['batch_idx']}

    @staticmethod
    def get_model_unrelated_attribute(trainer):
        return {'epoch': trainer.epoch, 'batch_idx': trainer.batch_idx}

    def save_states_to_pt(self, checkpoint_filename,
                          models, optimizers, trainer):
        checkpoint = {}
        for model in models:
            checkpoint[model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()

        model_unrelated_attribute = self.get_model_unrelated_attribute(trainer)
        for k in model_unrelated_attribute:
            checkpoint[k] = model_unrelated_attribute[k]

        model_related_attribute = self.get_model_related_attribute(trainer)
        for k in model_related_attribute:
            checkpoint[k] = model_related_attribute[k]

        torch.save(checkpoint, checkpoint_filename)
        return

    def save_checkpoint(self, models, optimizers, trainer):
        """Save checkpoint."""
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(
            os.path.join(self.save_dir, timestamp.strftime('%Y_%m_%d_%H_%M_%S') + '.pkl'))
        self.save_states_to_pt(
            checkpoint_filename, models, optimizers, trainer)

        logging.info(timestamp.strftime('%Y_%m_%d_%H_%M_%S') +
                   ' | Epoch: %d, Iteration %d' % (trainer.epoch, trainer.batch_idx))
        logging.info('Saving checkpoint file [' + checkpoint_filename + ']')
        return

    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        """Load a checkpoint."""
        if checkpoint_file is None:
            print('Loading checkpoint [' + self.checkpoint + ']')
            checkpoint_file = self.checkpoint
        checkpoint = torch.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                models[model].load_state_dict(checkpoint[model])
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        return {**self.get_model_unrelated_attribute_from_checkpoint(checkpoint), **self.get_model_related_attribute_from_checkpoint(checkpoint)}

    def get_checkpoint(self):
        """Get filename of checkpoint if it exists."""
        checkpoint_list = []
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pkl'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        self.checkpoint = None if (len(checkpoint_list) == 0) else checkpoint_list[-1]
        return

class BestFirstCheckpointSaver(ConstantCheckpointSaver):
    def get_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoint_list = []
        has_best = False
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename == 'best.pkl':
                    has_best = True
                elif filename.endswith('.pkl'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        if has_best:
            checkpoint_list.append(os.path.abspath(os.path.join(dirpath, 'best.pkl')))
        self.checkpoint = None if (len(checkpoint_list) == 0) else checkpoint_list[-1]
        return

    def save_checkpoint(self, models, optimizers, trainer):
        """Save checkpoint."""
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(
            os.path.join(self.save_dir, 'best.pkl'))
        self.save_states_to_pt(
            checkpoint_filename, models, optimizers, trainer)

        logging.info(timestamp.strftime('%Y_%m_%d_%H_%M_%S') +
                   ' | Epoch: %d, Iteration %d' % (trainer.epoch, trainer.batch_idx))
        logging.info('Saving checkpoint file [' + checkpoint_filename + ']')
        return
