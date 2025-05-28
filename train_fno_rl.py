import torch
import torch.optim as optim
from fno_model import FNOWindModel
from env_windfield import WindEnv
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FNOWindModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    env = WindEnv()

    for epoch in range(100):
        obs = env.reset()  # shape: [3, H, W]
        input_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        pred = model(input_tensor)
        pred_np = pred.squeeze(0).detach().cpu().numpy()

        reward = env.compute_reward(pred_np)
        loss = -torch.tensor(reward, requires_grad=True).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Reward: {reward:.6f}")

        if epoch % 10 == 0:
            plt.quiver(pred_np[0], pred_np[1])
            plt.title(f"Epoch {epoch}")
            plt.savefig(f"wind_epoch_{epoch}.png")
            plt.clf()

if __name__ == "__main__":
    train()
