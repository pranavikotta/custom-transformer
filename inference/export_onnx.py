'''
onnx (open neural network exchange) exports generated model into a static graph,
can be optimized, easily demonstrated, and run the model without the overhead of
the training environment.
'''
from pyexpat import model
from torch.mtia import device
import torch.onnx
import os

# Create a dummy input that matches your block_size
dummy_input = torch.zeros((1, block_size), dtype=torch.long).to(device)

# Export the model
onnx_path = os.path.join(drive_path, "transformer_model.onnx")
torch.onnx.export(model,
                  dummy_input,
                  onnx_path,
                  opset_version=14,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}})

print(f"Model exported to {onnx_path}!")