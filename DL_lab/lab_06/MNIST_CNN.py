import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from sklearn.metrics import confusion_matrix
import os

# Define transform for FashionMNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load FashionMNIST dataset
fashion_mnist_testset = datasets.FashionMNIST(root='.', train=False, download=False, transform=transform)
test_loader = DataLoader(fashion_mnist_testset, batch_size=64, shuffle=False)

# Define the CNN model (same as the one used for training)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classify_head = nn.Sequential(
            nn.Linear(64, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classify_head(features.reshape(-1, 64))  # Flatten the features to match the model's expectation

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.load_state_dict(torch.load('./ModelFiles/model.pt'))  # Load the trained weights
model.to(device)

# Print the model's state_dict (optional)
print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print()

# Evaluate the model on FashionMNIST
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_preds, all_labels = [], []

with torch.no_grad():  # Disable gradient calculation during inference
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class label

        all_preds.extend(predicted.cpu().numpy())  # Collect predictions
        all_labels.extend(labels.cpu().numpy())    # Collect actual labels

        correct += (predicted == labels).sum().item()  # Count correct predictions
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Overall Accuracy: {accuracy:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
