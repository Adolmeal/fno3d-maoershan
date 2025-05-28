import numpy as np

def generate_synthetic_field(grid_size=(16, 16, 16)):
    C, Z, Y, X = 4, *grid_size
    u = np.random.randn(Z, Y, X) * 0.1
    v = np.random.randn(Z, Y, X) * 0.1
    w = np.random.randn(Z, Y, X) * 0.1
    p = np.ones((Z, Y, X)) * 101325
    return np.stack([u, v, w, p], axis=0)

class WindEnv:
    def __init__(self, grid_size=(16, 16, 16)):
        self.grid_size = grid_size
        self.state = None

    def reset(self):
        base_field = np.random.randn(3, *self.grid_size) * 0.1  # 原始微弱扰动
        gust_field = np.zeros_like(base_field)
        gust_field[0] += 2.0  # x方向主风
        gust_field[1] += 1.0  # y方向次风
        gust_field += np.random.randn(*gust_field.shape) * 0.05  # 加少量扰动
        return base_field + gust_field

    def compute_reward(self, pred_field):
        u, v, w = pred_field[:3]
        try:
            div = (
                np.diff(u, axis=2, append=u[:, :, -1:]) +
                np.diff(v, axis=1, append=v[:, -1:, :]) +
                np.diff(w, axis=0, append=w[-1:, :, :])
            )
        except:
            div = np.zeros_like(u)
        reward = -np.mean(np.abs(div))
        return reward
