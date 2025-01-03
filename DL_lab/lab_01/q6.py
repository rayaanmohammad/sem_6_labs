import torch

random_tensor_1 = torch.randn(7, 7)
print(random_tensor_1)
print()

random_tensor_2 = torch.randn(1, 7)
print(random_tensor_1)
print()

random_tensor_3 = torch.matmul(random_tensor_1,random_tensor_2.T)
print(random_tensor_3)
print()