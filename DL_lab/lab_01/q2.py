import torch

tensor = torch.randn(2, 3, 4)
print("Original Tensor (Shape: 2x3x4):")
print(tensor)
print()
permuted_tensor = tensor.permute(2, 1, 0)
print("Permuted Tensor (Shape: 4x3x2):")
print(permuted_tensor)
print()

print("Shape of original tensor:", tensor.shape)
print("Shape of permuted tensor:", permuted_tensor.shape)
