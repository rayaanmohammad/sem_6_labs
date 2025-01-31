import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


X = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)

# Model setup
input_size = X.shape[1]
model = LogisticRegression(input_size)
print("Initial model parameters:", model.state_dict())

lr = 0.01
epochs = 1000

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

losses = []

# Training Loop
for epoch in range(epochs):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("BCE Loss")
plt.title("Training Loss Curve")
plt.show()


def classify(prob: torch.Tensor):
    return 0 if prob.item() < 0.5 else 1


test = torch.tensor([7], dtype=torch.float32).view(-1, 1)
model.eval()
with torch.inference_mode():
    pred = model(test)
    sigmoid_pred = torch.sigmoid(pred)
    print(f"Predicted value for input {test.item()} is {sigmoid_pred.item():.4f}")
    print(f"Class: {classify(sigmoid_pred)}")