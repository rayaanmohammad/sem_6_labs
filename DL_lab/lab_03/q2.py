import torch

x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

learning_rate = 0.001

epochs = 2

for epoch in range(epochs):
    y_pred = w * x + b
    loss = torch.mean((y_pred - y) ** 2)
    loss.backward()

    w_grad_analytical = torch.sum((y_pred - y) * x)
    b_grad_analytical = torch.sum(y_pred - y)

    print(f"Epoch {epoch + 1}:")
    print(f"PyTorch - w.grad = {w.grad.item()} | b.grad = {b.grad.item()}")
    print(f"Analytical - w_grad = {w_grad_analytical.item()} | b_grad = {b_grad_analytical.item()}")

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print(f"Updated w (PyTorch) = {w.item()}")
    print(f"Updated b (PyTorch) = {b.item()}")
    print(f"Loss (PyTorch) = {loss.item()}")

    w.grad.zero_()
    b.grad.zero_()

    print("-" * 40)