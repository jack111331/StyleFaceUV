import numpy as np
import torch
from operator import itemgetter
from sklearn.svm import SVC
from network.arch import Generator2D
from model.stylegan2_texture import StylecodeTo3DMMCoeffMLP

debug_output = False
example_num = 200000
pose_direction_save_path = "./data/pose_direction-test.pkl"
generator_2d_ckpt = './ckpt/StyleGAN2_ffhq_resol=256_550000.pt'
stylecode_to_3dmm_coeff_ckpt = './ckpt/stylecode_to_3dmm_coeff/lightning_logs/version_1/checkpoints/epoch=24-step=55674.ckpt'

'''
Referenced from "Interpreting the Latent Space of GANs for Semantic Face Editing" that there is a "Hyperplane boundary"
such that the stylecode "move" toward that direction, the specific semantics (e.g. Pose Semantic) would change at sudden.

We find the hyperplane its normal direction, so that we can control the pose of the image generated from our 
"manipulated stylecode" by moving original stylecode toward that hyperplane normal direction.
'''

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup model
    g_ema = Generator2D(256, 512, 8).to(device)
    g_ema.load_state_dict(torch.load(generator_2d_ckpt)["g_ema"], strict=False)
    stylecode_to_3dmm_coeff_model = StylecodeTo3DMMCoeffMLP(ckpt_path=stylecode_to_3dmm_coeff_ckpt).to(device)

    noise = torch.randn(example_num, 512).to(device)

    g_ema.eval()
    stylecode_to_3dmm_coeff_model.eval()
    wplus = g_ema.get_latent_Wplus(noise)

    yaw_angle_coeffs = np.zeros((example_num))
    batch_num = 50000
    for idx in range(wplus.shape[0] // batch_num):
        # Get yaw angle from its representing dimension in 3DMM coefficients (225: yaw angle)
        yaw_angle_coeffs[idx * batch_num: (idx+1) * batch_num] = stylecode_to_3dmm_coeff_model(torch.flatten(wplus[idx * batch_num: (idx+1) * batch_num], start_dim=1))[:, 225].detach().cpu().numpy()

    predictions = []
    for ind, item in enumerate(yaw_angle_coeffs):
        predictions.append([ind, item.item()])
    predictions = sorted(predictions, key=itemgetter(1))  # yaw angle (3DMM coeff) from small to large
    top_2_percent = int(example_num * 0.02)
    predictions_0 = np.array(predictions[:top_2_percent])
    predictions_1 = np.array(predictions[-top_2_percent:])

    predictions = np.concatenate((predictions_0, predictions_1), axis=0)
    labels = np.concatenate((np.array([0] * top_2_percent), np.array([1] * top_2_percent)), axis=0)

    clf = SVC(kernel='linear', C=1.0)
    W_space = []
    for i in range(len(predictions)):
        W_space.append(torch.flatten(wplus[int(predictions[i][0])]).detach().cpu().numpy())
    W_space = np.array(W_space)
    classifier = clf.fit(W_space, labels)
    pose_normal_direction = classifier.coef_ / np.linalg.norm(classifier.coef_)
    pose_normal_direction = torch.tensor(pose_normal_direction)
    torch.save(pose_normal_direction, pose_direction_save_path)
