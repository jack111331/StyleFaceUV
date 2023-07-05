styleGAN2_ckpt='./models/StyleGAN2_ffhq_resol=256_550000.pt'
latentCode_to_poseDirection_scalar_sym_ckpt='./checkpoints_latent2posescalar-new-sym/param.pkl'
latentCode_to_poseDirection_scalar_multi_pose_ckpt='./checkpoints_latent2posescalar-new_multi-pose/param.pkl'
styleCode_to_3DMMCoeff_ckpt='./checkpoints/sep_deeper/param22.pkl'

main_3D_model_predictor='./checkpoints_final/param.pkl'

import argparse

class Option:
    def __init__(self, profile='Common'):
        self.parser = argparse.ArgumentParser()

        com = self.parser.add_argument_group("Common")
        com.add_argument('--generator_2d_ckpt', type=str, default="./ckpt/StyleGAN2_ffhq_resol=256_550000.pt", help="path to the model checkpoint")
        com.add_argument('--generator_3d_ckpt', type=str, default='./ckpt/DispGAN/', help='Our final checkpoints')

        com.add_argument("--stylecode_3dmm_coeff_mlp_ckpt", type=str,
                         default='./ckpt/stylecode_3dmm_coeff_mlp/')

        com.add_argument("--stylecode_pose_scalar_mlp_ckpt", type=str,
                         default='./ckpt/stylecode_pose_scalar_mlp/')

        com.add_argument("--stylecode_multi_pose_scalar_mlp_ckpt", type=str,
                         default='./ckpt/stylecode_multi_pose_scalar_mlp/')

        com.add_argument("--size", type=int, default=256)
        com.add_argument("--latent", type=int, default=512)
        com.add_argument("--n_mlp", type=int, default=8)

        com.add_argument("--FACE_MODEL_PATH", type=str, default='./data/BFM_model_front.mat')
        com.add_argument("--FACE_UV_MODEL_PATH", type=str, default='./data/BFM_UV_refined.mat')
        com.add_argument("--coeff_yaw_angle_dim", type=int, default=225)

        com.add_argument("--dataset_dir", type=str, default="./data/")

        com.add_argument("--debug_output_dir", type=str, default="./debug/")
        com.add_argument("--debug_stylecode", type=str, default="./data/tensor-new.pkl")
        com.add_argument("--debug_3dmm_coeffs", type=str, default="./data/3DMMparam-new.pkl")
        com.add_argument("--debug_pose_normal_direction", type=str, default="./data/pose_direction-new.pkl")

        com.add_argument("--test_output_dir", type=str, default="./test/")

        if profile == "Style Code To 3DMM coeff MLP":
            tmp = self.parser.add_argument_group("Style Code To 3DMM coeff MLP")
            tmp.add_argument("--l_rate", type=float, default=1e-5)
            tmp.add_argument("--num_epochs", type=int, default=25)
            tmp.add_argument("--batch_size", type=int, default=16)
            com.add_argument("--ckpt_output_dir", type=str, default="./ckpt/stylecode_3dmm_coeff_mlp/")

        if profile == "Style Code To Multi Pose Direction MLP":
            tmp = self.parser.add_argument_group("Style Code To Multi Pose Direction MLP")
            tmp.add_argument("--l_rate", type=float, default=1e-3)
            tmp.add_argument("--num_epochs", type=int, default=25)
            tmp.add_argument("--batch_size", type=int, default=16)
            tmp.add_argument("--pose_normal_direction", type=str, default="./data/pose_direction-new.pkl")
            com.add_argument("--ckpt_output_dir", type=str, default="./ckpt/stylecode_multi_pose_scalar_mlp/")

        if profile == "Style Code To Pose Direction MLP":
            tmp = self.parser.add_argument_group("Style Code To Pose Direction MLP")
            tmp.add_argument("--l_rate", type=float, default=1e-3)
            tmp.add_argument("--num_epochs", type=int, default=25)
            tmp.add_argument("--batch_size", type=int, default=16)
            tmp.add_argument("--pose_normal_direction", type=str, default="./data/pose_direction-new.pkl")
            com.add_argument("--ckpt_output_dir", type=str, default="./ckpt/stylecode_pose_scalar_mlp/")

        if profile == "Pose Hyperplane Extraction":
            tmp = self.parser.add_argument_group("Style Code To Pose Direction MLP")
            tmp.add_argument("--Wplus", type=str, default='./data/tensor-new.pkl', help="path of W tensor.")
            tmp.add_argument("--exp_num", type=int, default=200000, help="Num of Examples taken to fit SVM")
            tmp.add_argument("--pose_direction_save_path", type=str, default="./data/pose_direction-new.pkl")

        if profile == "Disp GAN":
            tmp = self.parser.add_argument_group("Disp GAN")
            tmp.add_argument("--l_rate", type=float, default=1e-3)
            tmp.add_argument("--num_epochs", type=int, default=25)
            tmp.add_argument("--batch_size", type=int, default=12)
            tmp.add_argument("--n_critic", type=int, default=1)
            tmp.add_argument("--n_critic_d", type=int, default=5)
            tmp.add_argument("--photo_weight", type=float, default=10)
            tmp.add_argument("--d_reg_every", type=int, default=16)
            tmp.add_argument("--g_reg_every", type=int, default=4)
            com.add_argument("--ckpt_output_dir", type=str, default="./ckpt/DispGAN")

        if profile == "Disp GAN Single":
            tmp = self.parser.add_argument_group("Disp GAN Single")
            tmp.add_argument("--l_rate", type=float, default=1e-3)
            tmp.add_argument("--num_epochs", type=int, default=25)
            tmp.add_argument("--batch_size", type=int, default=12)
            tmp.add_argument("--n_critic", type=int, default=1)
            tmp.add_argument("--n_critic_d", type=int, default=5)
            tmp.add_argument("--photo_weight", type=float, default=10)
            tmp.add_argument("--d_reg_every", type=int, default=16)
            tmp.add_argument("--g_reg_every", type=int, default=4)
            com.add_argument("--ckpt_output_dir", type=str, default="./ckpt/DispGAN_single")

    def parse_args(self):
        self.args = self.parser.parse_args()
        return self.args
