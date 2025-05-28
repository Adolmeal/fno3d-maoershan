# reward.py
import torch
import torch.nn.functional as F

def divergence_loss(pred):
    # pred: [B,4,Z,Y,X]
    u,v,w,_ = pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    # central differences
    du_dx = u[:,:,:, :,1:] - u[:,:,:,:, :-1]
    dv_dy = v[:,:,:,1: , :] - v[:,:,:,:-1, :]
    dw_dz = w[:,:,1: ,: , :] - w[:,:-1,: ,: , :]
    # align to common cube
    D = du_dx[:,:,:,:-1] + dv_dy[:,:,:,:-1] + dw_dz[:,:,:-1]
    return torch.mean(D**2)

def energy_loss(pred):
    u,v,w,_ = pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    E = 0.5*(u**2+v**2+w**2)
    return torch.mean((E[:,1:-1,1:-1,1:-1] - E.mean())**2)

def total_reward(pred):
    return - (divergence_loss(pred) + 0.1*energy_loss(pred))
