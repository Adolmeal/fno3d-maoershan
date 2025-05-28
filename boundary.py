# boundary.py
import torch

def apply_boundary(pred, mask):
    """
    mask: [1,1,Z,Y,X] 二值墙面掩码，1=流体域，0=固壁
    将固壁速度置零，压力保持不变
    """
    u,v,w,p = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
    u = u * mask
    v = v * mask
    w = w * mask
    return torch.stack([u,v,w,p],dim=1)
