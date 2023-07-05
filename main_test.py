#!/usr/bin/python
"""
This script is a wrapper for the training process
You can view the full list of command line options by running `python train.py --help`.
The default values are the ones used to train the models in the paper.
Running the above command will start the training process. It will also create the folders `logs`
and `logs/train_example` that are used to save model checkpoints and Tensorboard logs.
If you start a Tensborboard instance pointing at the directory `logs` you should be able to look
at the logs stored during training.
"""
import torch
import numpy as np

from utils.options import Option
from evaluator import Evaluator

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    option = Option("Common")
    args = option.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(device, args)

    evaluator.test_pose_scalar()
    evaluator.test_multi_pose_scalar()
    # evaluator.test_stylegan2_to_3d_vanilla(100)
    # evaluator.test_stylegan2_to_3d_expression(100)
    # evaluator.test_stylegan2_to_3d_expression_variation_animate(107)
    # evaluator.test_stylegan2_to_3d_lighting(100)
    # evaluator.test_stylegan2_to_3d_lighting_variation(100)
    evaluator.test_stylegan2_to_3d_lighting_variation_animate(100)
    # evaluator.test_stylegan2_to_3d_interpolation(100)
    # evaluator.test_stylegan2_to_gradient_mask(100)
    # evaluator.test_fig3_both(100)
    # evaluator.test_fig3_paste(100)
