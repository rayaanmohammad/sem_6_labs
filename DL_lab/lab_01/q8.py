import torch

tensor_1 = torch.randn(2, 3)
tensor_2 = torch.randn(2, 3)

tensor_3 = torch.matmul(tensor_1,tensor_2.T)
print(tensor_3)