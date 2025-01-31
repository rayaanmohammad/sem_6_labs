import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Step 1: Define the Linear Regression Model by extending nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Model has only one weight (w) and one bias (b)
        self.w = nn.Parameter(torch.tensor(1.0))  # Initializing weight
        self.b = nn.Parameter(torch.tensor(1.0))  # Initializing bias

    def forward(self, x):
        # y_pred = w * x + b
        return self.w * x + self.b


# Step 2: Create a Custom Dataset for Linear Regression
class LinearRegressionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# Step 3: Generate some example data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).to('cuda')
y= torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).to('cuda')

# Step 4: Create the Dataset and DataLoader
dataset = LinearRegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Step 5: Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.001)  # SGD Optimizer

# Step 6: Training Loop
epochs = 100  # Number of epochs

for epoch in range(epochs):
    epoch_loss = 0.0  # To keep track of the loss for the epoch

    # Iterate through the data in batches using DataLoader
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # Zero the gradients before backpropagation

        # Forward pass: Compute predicted y
        y_pred = model(inputs)

        # Compute the loss
        loss = criterion(y_pred, targets)

        # Backward pass: Compute gradients
        loss.backward()

        # Update the model parameters using the optimizer
        optimizer.step()

        # Accumulate loss for monitoring
        epoch_loss += loss.item()

    # Print loss after each epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    # Print updated parameters
    print(f"Updated w = {model.w.item()}, b = {model.b.item()}")
    print("-" * 40)