# env_windfield.py
import torch
from data_gen import generate_synthetic_field
from reward import total_reward
from boundary import apply_boundary

class WindEnv:
    def __init__(self, grid_size=(32,32,32)):
        self.grid_size = grid_size
        self.mask = torch.ones((1,1,*grid_size))  # 默认全流体域，可定制为含固壁

    def reset(self):
        state = generate_synthetic_field(self.grid_size)  # [1,4,Z,Y,X]
        return state

    def step(self, pred):
        # pred: tensor [1,4,Z,Y,X]
        pred = apply_boundary(pred, self.mask)
        reward = total_reward(pred)
        return pred, reward
