# quick inspect: show keys and shapes
import torch
sd = torch.load('model_weights/your_vae.pth', map_location='cpu')
for k,v in sd.items():
    print(k, tuple(v.size()))
# or for a model instance:
model = BetaVAE()
for k,v in model.state_dict().items():
    print(k, tuple(v.size())