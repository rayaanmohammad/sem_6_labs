import torch

tensor = torch.arange(9).reshape(3, 3)
print("Original Tensor:")
print(tensor)
print()

reshaped_tensor = tensor.view(1, 9)
print("Reshaped Tensor (1x9):")
print(reshaped_tensor)
print()

viewed_tensor = reshaped_tensor.view(3, 3)
print("Viewed Tensor (3x3):")
print(viewed_tensor)
print()

stacked_tensor = torch.stack([tensor, tensor], dim=0)
print("Stacked Tensor (2 stacked 3x3 tensors):")
print(stacked_tensor)
print()

tensor_with_extra_dim = torch.rand(1, 3, 3, 1)
print("Tensor with Extra Dimension:")
print(tensor_with_extra_dim)
squeezed_tensor = tensor_with_extra_dim.squeeze()
print("Squeezed Tensor (removed size 1 dimensions):")
print(squeezed_tensor)
print()

unsqueezed_tensor = squeezed_tensor.unsqueeze(0)
print("Unsqueezed Tensor (adds a dimension at index 0):")
print(unsqueezed_tensor)
print()

unsqueezed_tensor_2 = squeezed_tensor.unsqueeze(2)
print("Unsqueezed Tensor at index 2 (adds a dimension at index 2):")
print(unsqueezed_tensor_2)
