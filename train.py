import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import HybridCFDModel
from env import WindEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs=100, save_path="checkpoints/hybrid_cfd_model.pt"):
    print(f"=== Starting training on device: {device} ===")

    # 自动创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = WindEnv(grid_size=(16, 16, 16))
    model = HybridCFDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        state = env.reset()  # shape: [C, D, H, W]
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # [1, C, D, H, W]

        pred = model(input_tensor)  # [1, C, D, H, W]
        reward = env.compute_reward(pred[0].detach().cpu().numpy())

        # 假设前三通道为监督目标
        loss = loss_fn(pred, input_tensor[:, :3])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.5f}, Reward={reward:.5f}")

    # 保存“完整模型”（结构 + 参数）
    torch.save(model, save_path)
    print(f"\n✅ 完整模型已保存至: {save_path}")

if __name__ == "__main__":
    train()
