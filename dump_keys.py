import torch
import json
from rawnet2_model import RawNet

with open("model_config_RawNet.json", "r") as f:
    d_args = json.load(f)

model = RawNet(d_args, 'cpu')
state_dict = torch.load("pre_trained_DF_RawNet2.pth", map_location="cpu")
for k, v in state_dict.items():
    print(k, list(v.shape))
