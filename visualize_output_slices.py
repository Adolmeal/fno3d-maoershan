# visualize_output_slices.py
import torch
import matplotlib.pyplot as plt
from env import WindEnv

# 加载完整模型（结构+参数）
model = torch.load(
    "checkpoints/hybrid_cfd_model.pt",
    map_location="cpu",
    weights_only=False
)
model.eval()

env = WindEnv(grid_size=(16, 16, 16))
state = env.reset()  # [C, D, H, W]
input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, C, D, H, W]

with torch.no_grad():
    pred = model(input_tensor)[0]  # [C, D, H, W]

# 中间平面切片可视化
mid_z = pred.shape[1] // 2
u = pred[0, mid_z].numpy()
v = pred[1, mid_z].numpy()
p = pred[2, mid_z].numpy()

Y, X = torch.meshgrid(torch.arange(u.shape[0]), torch.arange(u.shape[1]), indexing='ij')
plt.figure(figsize=(6,5))
plt.quiver(X, Y, u, v, scale=1)
plt.title("Predicted Velocity Field (mid-plane)")
plt.savefig("output_velocity_quiver.png")
plt.close()

plt.figure(figsize=(6,5))
plt.imshow(p, cmap='coolwarm')
plt.colorbar(label='Pressure')
plt.title("Predicted Pressure Field (mid-plane)")
plt.savefig("output_pressure_heatmap.png")
plt.close()

print("✅ 已保存 output_velocity_quiver.png 以及 output_pressure_heatmap.png")

