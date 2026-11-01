import torch
import ai_edge_torch
from rawnet2_model import RawNet
import json

print("Initializing RawNet2 model...")
with open("model_config_RawNet.json", "r") as f:
    d_args = json.load(f)

model = RawNet(d_args, 'cpu')

print("Loading weights...")
state_dict = torch.load("pre_trained_DF_RawNet2.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Converting using ai-edge-torch...")
dummy_input = torch.randn(1, 64600)

edge_model = ai_edge_torch.convert(model, (dummy_input,))
edge_model.export("rawnet2_float32.tflite")
print("Conversion complete! Output saved to rawnet2_float32.tflite")
