# visualize_feature_maps.py
import torch
import matplotlib.pyplot as plt

# 先允许反序列化整个模型
model = torch.load(
    "checkpoints/hybrid_cfd_model.pt",
    map_location="cpu",
    weights_only=False
)
model.eval()

# 钩子收集每个 Conv3d 层的特征
feature_maps = {}
def hook_fn(module, inp, out):
    feature_maps[module] = out.detach()[0, 0].cpu()

for module in model.modules():
    if isinstance(module, torch.nn.Conv3d):
        module.register_forward_hook(hook_fn)

# 构造假输入
dummy_input = torch.randn(1, 3, 16, 16, 16)
with torch.no_grad():
    _ = model(dummy_input)

# 可视化每个 Conv3d 层的中间特征切片
mid_z = dummy_input.shape[2] // 2
for i, (module, fmap) in enumerate(feature_maps.items()):
    slice_img = fmap[mid_z].numpy()
    plt.imshow(slice_img, cmap='viridis')
    plt.title(f"Layer {i+1}: {module.__class__.__name__}")
    plt.colorbar()
    plt.savefig(f"feature_map_{i+1}.png")
    plt.clf()
    print(f"✅ Saved feature_map_{i+1}.png")

