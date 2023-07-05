import numpy as np
import argparse
import os
import torch
from operator import itemgetter
from sklearn.svm import SVC
from network.arch import Generator2D, StylecodeTo3DMMCoeffMLP
import time
def get_arguments():
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("--Wplus", type=str, default='./pkls/tensor-new.pkl', help="path of W tensor.")
    parser.add_argument("--exp_num", type=int, default=200000, help="Num of Examples taken to fit SVM")
    parser.add_argument("--load_model",type=str,default='./checkpoints/sep_deeper/param22.pkl')
    return parser.parse_args()
def main():
    args = get_arguments()
    start_time = time.time()
    g_ema= Generator2D(256, 512, 8, channel_multiplier=2).cuda()
    g_ema.load_state_dict(torch.load("./models/StyleGAN2_ffhq_resol=256_550000.pt")["g_ema"],strict=False)
    g_ema.eval()
    model = StylecodeTo3DMMCoeffMLP().cuda()
    model.load_state_dict(torch.load(args.load_model))
    model.eval()
    noise = torch.randn(args.exp_num,512).cuda()
    wplus = g_ema.get_latent_Wplus(noise)
    coeffs = model(torch.flatten(wplus, start_dim=1))
    # 225 是 3DMM coefficient中
    coeffs = coeffs[:,225] # yaw angle: Reference from "Interpreting the Latent Space of GANs for Semantic Face Editing"???
    print(wplus.size(), coeffs.size())
    predictions = []
    for ind, item in enumerate(coeffs):
        predictions.append([ind, item.item()])
    predictions = sorted(predictions, key=itemgetter(1)) ## yaw angle from low to high
    #print(predictions[0])
    predictions_0 = predictions[:int(args.exp_num*0.02)] # 取TOP 2%標
    predictions_1 = predictions[-1*int(args.exp_num*0.02):]
    print(len(predictions_0))
    predictions = np.array(predictions_0+predictions_1)
    labels = np.array([0]*int(args.exp_num*0.02)+[1]*int(args.exp_num*0.02))
    clf = SVC(kernel='linear', C = 1.0)
    W_space = []
    for i in range(len(predictions)):
        W_space.append(torch.flatten(wplus[int(predictions[i][0])]).detach().cpu().numpy())
    W_space = np.array(W_space)
    print(W_space.shape)
    classifier = clf.fit(W_space,labels)
    boundary = classifier.coef_ / np.linalg.norm(classifier.coef_)
    print(classifier.coef_.shape)  #(1, 512*14)
    boundary = torch.tensor(boundary)
    torch.save(boundary,"./pose-dir.pkl")
    print(time.time()-start_time)

    

main()
