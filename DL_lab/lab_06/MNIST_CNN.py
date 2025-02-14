import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch import nn
from sklearn.metrics import accuracy_score

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
train = datasets.MNIST('.', train= True, download= False, transform= transforms)
test = datasets.MNIST('.', train= False, download= False, transform= transforms)

train_loader = DataLoader(train, batch_size= 64, shuffle= True)
test_loader = DataLoader(test, batch_size= 64, shuffle= True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride= 2),
                                 nn.Conv2d(32, 64, kernel_size=3), 
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(64, 32, kernel_size= 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride= 2))
        self.classify_head = nn.Sequential(nn.Linear(32, 20, bias= True),
                                           nn.Linear(20, 10, bias= True))
    
    def forward(self, x):
        return self.classify_head(self.net(x).reshape(-1, 32))

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=device)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], loss = {running_loss}')

torch.save(model, 'model.pt')

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix,"\n")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")

print("Accuracy score of the model: ",accuracy_score(y_true, y_pred))