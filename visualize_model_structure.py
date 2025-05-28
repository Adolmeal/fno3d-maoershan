# visualize_model_structure.py
import torch
from torchviz import make_dot
from model import HybridCFDModel

# 1. 显式允许加载整个模型结构（weights_only=False）
model = torch.load(
    "checkpoints/hybrid_cfd_model.pt",
    map_location="cpu",
    weights_only=False          # <— 关键：加载结构+参数
)
model.eval()

# 2. 构造假输入，请根据你的实际通道数和网格大小调整
dummy_input = torch.randn(1, 3, 16, 16, 16)  # [B, C, D, H, W]

# 3. 前向一次
output = model(dummy_input)
if isinstance(output, tuple):
    output = output[0]

# 4. 生成并保存计算图
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("hybrid_cfd_model_structure")

print("✅ 模型结构图已生成：hybrid_cfd_model_structure.png")
