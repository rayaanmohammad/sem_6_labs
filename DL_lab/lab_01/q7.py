import torch

tensor_1 = torch.randn(2, 3)
tensor_2 = torch.randn(2, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)

print("Tensor 1:")
print(tensor_1)
print()

print("Tensor 2:")
print(tensor_2)
