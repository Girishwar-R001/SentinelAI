import torch
import ai_edge_torch
import json
from rawnet2_model import RawNet

d_args = {
    'nb_samp': 64600,
    'first_conv': 129,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_classes': 2
}

model = RawNet(d_args, 'cpu')
state_dict = torch.load("pre_trained_DF_RawNet2.pth", map_location="cpu", weights_only=False)
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 64600, dtype=torch.float32)

print("Converting to TFLite with ai-edge-torch...")
edge_model = ai_edge_torch.convert(model, (dummy_input,))
edge_model.export("rawnet2_edge.tflite")
print("Exported to rawnet2_edge.tflite!")
