import torch
import matplotlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
test = datasets.FashionMNIST('.', train= False, transform= transforms, download= True)
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
torch.serialization.add_safe_globals([CNN, nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear])
model = torch.load('model.pt')
model.to('cuda')

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        output = model(input)
        val, index = torch.max(output, dim= 1)
        all_preds.extend(index.to('cpu'))
        all_labels.extend(target.to('cpu'))
print(accuracy_score(all_preds, all_labels))