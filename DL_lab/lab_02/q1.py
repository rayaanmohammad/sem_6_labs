import torch

a = torch.tensor([1.5], requires_grad=True)
b = torch.tensor([2.5], requires_grad=True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()
print("Pytorch gradient:")
print("dz/da:",a.grad)
print("dz/db:",b.grad)
print()

def manual_gradient_computation(a, b):
    x = 2 * a + 3 * b
    y = 5 * a * a + 3 * b * b * b
    z = 2 * x + 3 * y

    dz_dx = 2
    dz_dy = 3
    dx_da = 2
    dx_db = 3
    dy_da = 10 * a
    dy_db = 9 * b * b
    dz_da = dz_dx * dx_da + dz_dy * dy_da
    dz_db = dz_dx * dx_db + dz_dy * dy_db
    return dz_da, dz_db

a_val = 1.5
b_val = 2.5
grad_a, grad_b = manual_gradient_computation(a_val, b_val)
print("Analytical gradient:")
print("dz/da:", grad_a)
print("dz/db:", grad_b)
