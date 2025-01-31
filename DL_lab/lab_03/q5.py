import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Step 1: Define the Linear Regression Model using nn.Linear
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # nn.Linear(in_features, out_features), input size=1, output size=1 for simple linear regression
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Step 2: Input data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).view(-1,
                                                                                    1)  # reshape for single feature input
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).view(-1, 1)  # reshape for output

# Step 3: Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.001)  # SGD optimizer

# Step 4: Training Loop
epochs = 100
losses = []  # To store the loss after each epoch for plotting

for epoch in range(epochs):
    # Zero the gradients before the backward pass
    optimizer.zero_grad()

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Backward pass: Compute gradients
    loss.backward()

    # Update the parameters using the optimizer
    optimizer.step()

    # Store the loss for plotting
    losses.append(loss.item())

    # Optionally print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Plotting the Loss vs Epochs
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs for Linear Regression')
plt.grid(True)
plt.show()

# Step 6: Visualize the learned line
with torch.no_grad():
    predicted = model(x)

plt.scatter(x.numpy(), y.numpy(), color='blue', label='Original Data')
plt.plot(x.numpy(), predicted.numpy(), color='red', label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with nn.Linear()')
plt.legend()
plt.grid(True)
plt.show()

# Print the learned parameters (w and b)
print(f'Learned weight (w): {model.linear.weight.item()}')
print(f'Learned bias (b): {model.linear.bias.item()}')