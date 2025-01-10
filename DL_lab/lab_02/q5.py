import torch

x = torch.tensor([1.5], requires_grad=True)
y = 8*x**4 + 3*x**3 + 7*x**2 + 6*x + 3

y.backward()
print("Pytorch gradient:")
print("dy/dx:", x.grad)
print()

def manual_gradient_computation(x):
    dy_dx = 32*x**3 + 9*x**2 + 14*x + 6
    return dy_dx


x_val = 1.5
grad_x = manual_gradient_computation(x_val)
print("Analytical gradient:")
print("dy/dx:", grad_x)