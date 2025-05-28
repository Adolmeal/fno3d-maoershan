import torch

model_path = 'checkpoints/hybrid_cfd_model.pt'
loaded = torch.load(model_path, map_location='cpu')

if isinstance(loaded, dict):
    print("✅ This .pt file contains only the state_dict (model parameters).")
else:
    print("✅ This .pt file contains the full model (structure + parameters).")
