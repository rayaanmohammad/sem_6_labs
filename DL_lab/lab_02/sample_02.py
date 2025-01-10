import torch

x=torch.tensor(3.5,requires_grad=True)
y=x*x
z=2*y+3

print("x:",x)
print("y=x*x:",y)
print("z=2*y:",z)

# work out gradients
z.backward()
print("Working out gradients dz/dx")
# what is gradient at x = 3.5
print("Gradient at x = 3.5: ", x.grad)