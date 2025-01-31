import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X1 = torch.tensor([3, 4, 5, 6, 2], dtype=torch.float32).view(-1, 1)
X2 = torch.tensor([8, 5, 7, 3, 1], dtype=torch.float32).view(-1, 1)
y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7],dtype=torch.float32).view(-1, 1)

X = torch.concatenate((X1, X2), dim=-1)

print(X.shape, y.shape)

class MultipleLinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)

input_size = X.shape[1]
output_size = y.shape[1]

model = MultipleLinearRegression(input_size, output_size)
print(model.state_dict())

lr = 0.001
epochs = 1000

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

losses = []

for epoch in range(epochs):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss}")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()

test = torch.tensor([3, 2], dtype=torch.float32)
model.eval()
with torch.inference_mode():
    pred = model(test)
    print(f"Predicted value for input {test} is {pred}")