import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

bs = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.001, save_path="./models/model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        validate(model, test_loader, device)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Current Accuracy: {accuracy:.2f}%")
    return accuracy

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)        
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CIFARCNN(nn.Module):
    def __init__(self):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_train_data = datasets.MNIST(root='./raw_data', train=True, download=True, transform=transform)
mnist_test_data = datasets.MNIST(root='./raw_data', train=False, transform=transform)

mnist_train_loader = DataLoader(mnist_train_data, batch_size=bs, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_data, batch_size=bs, shuffle=False)

cifar_train_set = datasets.CIFAR10(root='./raw_data', train=True, download=True, transform=transform)
cifar_test_set = datasets.CIFAR10(root='./raw_data', train=False, download=True, transform=transform)

cifar_train_loader = DataLoader(cifar_train_set, batch_size=128, shuffle=True, num_workers=2)
cifar_test_loader = DataLoader(cifar_test_set, batch_size=128, shuffle=False, num_workers=2)

MnistCNN = MNISTCNN().to(device)
train_model(model = MnistCNN, train_loader = mnist_train_loader, test_loader = mnist_test_loader, device = device, epochs = 5, save_path = "./models/mnist_cnn.pth")

cifarCNN = CIFARCNN().to(device)
train_model(model = cifarCNN, train_loader = cifar_train_loader, test_loader = cifar_test_loader, device = device, epochs = 10, save_path = "./models/cifar_resnet18.pth")
