# data_gen.py
import numpy as np
import torch

def generate_synthetic_field(grid_size=(32,32,32), wind_speed=5.0):
    """
    返回 shape = [1, 4, Z, Y, X]  的 tensor：
      通道0-2: u,v,w; 通道3: pressure p
    """
    Z,Y,X = grid_size
    z = np.linspace(0,1,Z)
    y = np.linspace(0,1,Y)
    x = np.linspace(0,1,X)
    Zg,Yg,Xg = np.meshgrid(z,y,x,indexing='ij')
    # 简单旋涡+层流叠加
    u =  wind_speed * np.sin(np.pi*Xg)*np.cos(np.pi*Yg)*np.exp(-Zg)
    v = -wind_speed * np.cos(np.pi*Xg)*np.sin(np.pi*Yg)*np.exp(-Zg)
    w = 0.1 * np.sin(2*np.pi*Zg)
    p = 1e5 + 1000*(Zg-0.5)
    data = np.stack([u,v,w,p],axis=0)[None,...]
    return torch.tensor(data, dtype=torch.float32)
