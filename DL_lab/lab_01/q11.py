import torch

torch.manual_seed(7)

tensor = torch.randn(1, 1, 1, 7)

print("Original Tensor:",tensor)
print()

squeezed_tensor = tensor.squeeze()
print("Squeezed Tensor:",squeezed_tensor)
