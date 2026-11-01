import torch
from rawnet2_model import RawNet
import onnx

# The exact d_args for the ASVspoof 2021 RawNet2 baseline
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

print("Initializing RawNet2 model...")
model = RawNet(d_args, 'cpu')

print("Loading weights...")
state_dict = torch.load("pre_trained_DF_RawNet2.pth", map_location="cpu", weights_only=False)
model.load_state_dict(state_dict)
model.eval()

print("Exporting to ONNX...")
dummy_input = torch.randn(1, 64600, dtype=torch.float32)
torch.onnx.export(model, dummy_input, "rawnet2.onnx", 
                  export_params=True, 
                  opset_version=13, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output'])

print("ONNX export complete. Now run: onnx2tf -i rawnet2.onnx")
