import torch

x=torch.rand(3,requires_grad=True)
print(x)
print()

y=torch.rand(3)
y.requires_grad=True
print(y)