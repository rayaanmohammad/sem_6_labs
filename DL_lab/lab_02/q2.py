import torch

x = torch.tensor([1.5], requires_grad=True)
w = torch.tensor([2.5], requires_grad=True)
b = torch.tensor([3.5], requires_grad=True)

u=w*x
v=u+b
a=torch.nn.functional.relu(v)

a.backward()
print("Pytorch gradient:")
print("da/dx:",x.grad)
print("da/dw:",w.grad)
print("da/db:",b.grad)
print()


def manual_relu_gradient(x, w, b):
    u = w * x
    v = u + b

    # ReLU activation condition
    if v > 0:
        da_dv = 1
    else:
        da_dv = 0

    dv_du = 1
    dv_db = 1
    du_dx = w
    du_dw = x

    da_dx = da_dv * dv_du * du_dx
    da_dw = da_dv * dv_du * du_dw
    da_db = da_dv * dv_db
    return da_dx, da_dw, da_db

x_val = 1.5
w_val = 2.5
b_val = 3.5

grad_x, grad_w, grad_b = manual_relu_gradient(x_val, w_val, b_val)
print("Analytical gradient:")
print("da/dx:", grad_x)
print("da/dw:", grad_w)
print("da/db:", grad_b)
