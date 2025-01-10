import torch
import numpy as np

x = torch.tensor([1.5], requires_grad=True)
y = torch.tensor([2.5], requires_grad=True)
z = torch.tensor([3.5], requires_grad=True)

w = torch.tanh(torch.log(1+z*((2*x)/torch.sin(y))))

w.backward()
print("Pytorch gradient:")
print("df/dx:", x.grad)
print("df/dw:", y.grad)
print("df/db:", z.grad)
print()

def manual_gradient_computation(x, y, z):
    a = 2 * x
    b = torch.sin(y)
    c = a / b
    d = z * c
    e = torch.log(1 + d)
    f = torch.tanh(e)

    da_dx = 2
    db_dy = torch.cos(y)
    dc_da = 1 / b
    dc_db = -a / b**2
    dd_dc = z
    dd_dz = c
    de_dd = 1 / (1 + d)
    df_de = 1 - torch.tanh(e)**2

    df_dx = df_de * de_dd * dd_dc * dc_da * da_dx
    df_dy = df_de * de_dd * dd_dc * dc_db * db_dy
    df_dz = df_de * de_dd * dd_dz

    return df_dx,df_dy,df_dz

grad_x, grad_y, grad_z = manual_gradient_computation(x, y, z)
print("Analytical gradient:")
print("dw/dx:", grad_x)
print("dw/dy:", grad_y)
print("dw/dz:", grad_z)