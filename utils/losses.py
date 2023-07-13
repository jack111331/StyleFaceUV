import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import cv2
# from op import conv2d_gradfix
import torch.autograd as autograd
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
def photo_loss(pred_img, gt_img, img_mask):
    pred_img = pred_img * 255
    gt_img = gt_img * 255
    pred_img = pred_img.float()
    loss = torch.sqrt(torch.sum(torch.square(pred_img - gt_img), 3)+1e-8 )*img_mask/255
    loss = torch.sum(loss, dim=(1, 2)) / (torch.sum(img_mask, dim=(1, 2)))
    #print("Loss : ",torch.sum(img_mask).item())
    loss = torch.mean(loss)
    #print(loss.item())
    # loss = torch.sum(loss)/torch.sum(img_mask)

    return loss

def lm_loss(pred_lms, gt_lms, img_size=256):
    w = torch.ones((1, 68)).to(pred_lms.device)
    #w[:, 28:31] = 10
    #w[:, 48:68] = 10
    w[:, 28:31] = 50
    w[:, 48:68] = 50
    norm_w = w / w.sum()
    loss = torch.sum(torch.square(pred_lms/img_size - gt_lms/img_size), dim=2) * norm_w
    loss = torch.mean(loss.sum(1))

    return loss

def reg_loss(id_coeff, ex_coeff, tex_coeff):
    loss = torch.square(id_coeff).sum() + \
            torch.square(tex_coeff).sum() *1.7e-3 + \
            torch.square(ex_coeff).sum(1).mean() * 0.8

    return loss
def reg_loss_tex(tex_coeff):
    loss = torch.square(tex_coeff).sum() *1.7e-3
    return loss

def reflectance_loss(tex, skin_mask):

    skin_mask = skin_mask.unsqueeze(2)
    tex_mean = torch.sum(tex*skin_mask, 1, keepdims=True)/torch.sum(skin_mask)
    loss = torch.sum(torch.square((tex-tex_mean)*skin_mask/255.))/ \
        (tex.shape[0]*torch.sum(skin_mask))

    return loss

def gamma_loss(gamma):

    gamma = gamma.reshape(-1, 3, 9)
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

    return gamma_loss



def l1loss(pred,gt):
    return torch.mean(torch.abs(pred-gt)) #+ 0.1*torch.mean(torch.abs(pred[:,144:224]-gt[:,144:224]))
def l2loss(pred,gt):
    return torch.sqrt(torch.sum(torch.square(pred-gt)))


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()
def d_r1_loss(real_pred, real_img):
    #with conv2d_gradfix.no_weight_gradients():
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty
