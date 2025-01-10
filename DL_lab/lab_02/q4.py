import torch
import numpy as np

x = torch.tensor([1.5], requires_grad=True)
y = torch.exp(-(x*x)-(2*x)-torch.sin(x))

y.backward()
print("Pytorch gradient:")
print("dy/dx:", x.grad)
print()

def manual_gradient_computation(x):
    a = x * x
    b = 2 * x
    c = np.sin(x)
    y = np.exp(-(a + b + c))

    da_dx = 2 * x
    db_dx = 2
    dc_dx = np.cos(x)
    dy_dx = y * (-(da_dx + db_dx + dc_dx))

    return dy_dx


x_val = 1.5
grad_x = manual_gradient_computation(x_val)
print("Analytical gradient:")
print("dy/dx:", grad_x)