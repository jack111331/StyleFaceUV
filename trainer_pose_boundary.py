import numpy as np
import torch
from operator import itemgetter
from sklearn.svm import SVC
from network.arch import Generator2D
from network.arch import StylecodeTo3DMMCoeffMLP
from utils.options import Option
from trainer_3dcoeff import BestFirst3DCoeffCheckpointSaver

debug_output = False

'''
Referenced from "Interpreting the Latent Space of GANs for Semantic Face Editing" that there is a "Hyperplane boundary"
such that the stylecode "move" toward that direction, the specific semantics (e.g. Pose Semantic) would change at sudden.

We find the hyperplane its normal direction, so that we can control the pose of the image generated from our 
"manipulated stylecode" by moving original stylecode toward that hyperplane normal direction.
'''

if __name__ == '__main__':
    option = Option("Pose Hyperplane Extraction")
    args = option.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup model
    g_ema = Generator2D(args.size, args.latent, args.n_mlp).to(device)
    g_ema.load_state_dict(torch.load(args.generator_2d_ckpt)["g_ema"], strict=False)
    stylecode_to_3dmm_coeff_model = StylecodeTo3DMMCoeffMLP().to(device)
    stylecode_to_3dmm_saver = BestFirst3DCoeffCheckpointSaver(args.stylecode_3dmm_coeff_mlp_ckpt)
    if stylecode_to_3dmm_saver.checkpoint != None:
        stylecode_to_3dmm_saver.load_checkpoint(
            models={"Stylecode to 3DMM Coeff": stylecode_to_3dmm_coeff_model},
            optimizers={})
    noise = torch.randn(args.exp_num, 512).to(device)

    g_ema.eval()
    stylecode_to_3dmm_coeff_model.eval()
    wplus = g_ema.get_latent_Wplus(noise)
    coeffs = stylecode_to_3dmm_coeff_model(torch.flatten(wplus, start_dim=1))
    # Get yaw angle from its representing dimension in 3DMM coefficients
    coeffs = coeffs[:, args.coeff_yaw_angle_dim]

    predictions = []
    for ind, item in enumerate(coeffs):
        predictions.append([ind, item.item()])
    predictions = sorted(predictions, key=itemgetter(1))  # yaw angle (3DMM coeff) from small to large
    top_2_percent = int(args.exp_num * 0.02)
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
    torch.save(pose_normal_direction, args.pose_direction_save_path)
