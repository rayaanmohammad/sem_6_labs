import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).view(-1, 1)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(2, 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        return x

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

full_dataset = MyDataset(X, Y)
batch_size = 1
train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XORModel().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 10000
loss_list = []

for epoch in range(epochs):
    total_loss = 0

    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_data_loader)
    loss_list.append(avg_loss)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss}')

for name, param in model.named_parameters():
    print(f'{name}: {param.data}')

with torch.no_grad():
    for inputs in X:
        inputs = inputs.to(device)
        prediction = model(inputs)
        print(f"Input: {inputs.cpu().numpy()}, Prediction: {prediction.cpu().item()}")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")

plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss curve during training')
plt.show()
