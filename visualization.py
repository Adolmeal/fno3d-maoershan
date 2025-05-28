# visualization.py
import torch
import matplotlib.pyplot as plt
from model import HybridCFDModel
from env import WindEnv
import numpy as np

def infer_and_plot(ckpt="checkpoints/hybrid_model.pt"):
    device = torch.device("cpu")
    model = HybridCFDModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    env = WindEnv(grid_size=(16, 16, 16))
    state = env.reset()
    input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0]  # [3, D, H, W]

    # 中间Z平面切片
    mid_z = pred.shape[1] // 2
    u = pred[0, mid_z].cpu().numpy()
    v = pred[1, mid_z].cpu().numpy()
    mag = np.sqrt(u**2 + v**2)

    X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    plt.figure(figsize=(6, 5))
    plt.quiver(X, Y, u, v, mag, scale=1, scale_units='xy', cmap='jet')
    plt.colorbar(label="Velocity Magnitude")
    plt.title("Mid-plane Velocity Field (HybridCFDModel)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    infer_and_plot()
