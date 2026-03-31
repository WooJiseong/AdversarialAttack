import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

os.makedirs("./results", exist_ok=True)
bs = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기본 Train / Test 함수
def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.003, save_path="./models/model.pth"):
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

# 대상 모델 (Mnist / Cifar10)
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
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 데이터 로드 및 처리
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

# 모델 초기화 및 학습
MnistCNN = MNISTCNN().to(device)
train_model(model = MnistCNN, train_loader = mnist_train_loader, test_loader = mnist_test_loader, device = device, epochs = 5, save_path = "./models/mnist_cnn.pth")

cifarCNN = CIFARCNN().to(device)
train_model(model = cifarCNN, train_loader = cifar_train_loader, test_loader = cifar_test_loader, device = device, epochs = 15, save_path = "./models/cifar_resnet18.pth")

# 공격 기법 함수
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    outputs = model(x_adv)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, target)

    model.zero_grad()
    loss.backward()
    data_grad = x_adv.grad.data
    sign_data_grad = data_grad.sign()
    
    x_adv = x_adv - eps * sign_data_grad
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def fgsm_untargeted(model, x, label, eps):
    input_tensor = x.clone().detach()
    input_tensor.requires_grad = True

    outputs = model(input_tensor)
    
    loss = F.cross_entropy(outputs, label)

    model.zero_grad()
    loss.backward()

    if input_tensor.grad is None:
        print("Error: Gradient is None!")
        return x
        
    sign_data_grad = input_tensor.grad.sign()

    adv_img = input_tensor + eps * sign_data_grad
    adv_img = torch.clamp(adv_img, 0, 1)

    return adv_img.detach()

def pgd_targeted(model, x, target, k, eps, eps_step):

    x_adv = x.clone().detach()
    
    for i in range(k):
        x_adv.requires_grad = True
        
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, target)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.detach()
        x_adv = x_adv - eps_step * grad.sign()
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, min=0, max=1).detach()
        
    return x_adv

def pgd_untargeted(model, x, label, k, eps, eps_step):

    x_adv = x.clone().detach()
    
    for i in range(k):
        x_adv.requires_grad = True
        
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, label)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.detach()
        x_adv = x_adv + eps_step * grad.sign()
        
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, min=0, max=1).detach()
        
    return x_adv

# 공격 성공률 연산 및 결과 시각화
def attack_visualization(orig, adv, orig_label, adv_label, attack_name, dataset_name, eps, idx):
    orig_np = orig.squeeze().cpu().detach().numpy()
    adv_np = adv.squeeze().cpu().detach().numpy()
    
    perturbation = adv_np - orig_np
    
    pert_display = 0.5 + perturbation
    pert_display = np.clip(pert_display, 0, 1)

    if len(orig_np.shape) == 3:
        orig_display = np.transpose(orig_np, (1, 2, 0))
        adv_display = np.transpose(adv_np, (1, 2, 0))
        pert_display = np.transpose(pert_display, (1, 2, 0))
    else:
        orig_display = orig_np
        adv_display = adv_np
        pert_display = pert_display

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(orig_display, cmap='gray' if len(orig_np.shape)==2 else None)
    axes[0].set_title(f"Original (Label: {orig_label})")
    axes[0].axis('off')
    
    axes[1].imshow(adv_display, cmap='gray' if len(orig_np.shape)==2 else None)
    axes[1].set_title(f"Adversarial (Pred: {adv_label})")
    axes[1].axis('off')
    
    axes[2].imshow(pert_display, cmap='gray' if len(orig_np.shape)==2 else None)
    axes[2].set_title(f"True Perturbation (Eps: {eps})")
    axes[2].axis('off')
    
    plt.tight_layout()
    filename = f"./results/{dataset_name}_{attack_name}_eps{eps}_sample{idx}.png"
    plt.savefig(filename)
    plt.close()

def attack_rate(model, loader, device, attack_func, attack_name, dataset_name, eps_list=[0.05, 0.1, 0.2, 0.3]):
    model.eval()
    
    print(f"\n>>> Running Benchmark: {attack_name} on {dataset_name}")
    print(f"{'Epsilon':<10} | {'Success Rate':<15} | {'Successful/Total'}")
    print("-" * 50)

    for eps in eps_list:
        success_count = 0
        total_count = 0
        visualized_count = 0
        
        for images, labels in loader:
            if total_count >= 150:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pre_preds = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if total_count >= 150: break
                
                if pre_preds[i] != labels[i]:
                    continue
                
                total_count += 1
                img, label = images[i:i+1], labels[i:i+1]
                
                is_targeted = "untargeted" not in attack_name.lower()
                target = (label + 1) % 10 if is_targeted else label
                
                adv_img = attack_func(model, img, target, eps)
                adv_output = model(adv_img)
                _, post_pred = torch.max(adv_output, 1)
                
                is_success = False
                if is_targeted:
                    if post_pred.item() == target.item():
                        is_success = True
                else:
                    if post_pred.item() != label.item():
                        is_success = True
                
                if is_success:
                    success_count += 1
                    if visualized_count < 7:
                        attack_visualization(img, adv_img, label.item(), post_pred.item(), 
                                           attack_name, dataset_name, eps, visualized_count)
                        visualized_count += 1

        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        print(f"{eps:<10} | {success_rate:>11.2f}% | {success_count:>4}/{total_count}")

    print("-" * 50)

# 시각화 시작

attacks = {
    "Untargeted_FGSM": fgsm_untargeted,
    "Targeted_FGSM": fgsm_targeted,
    "Untargeted_PGD": lambda m, x, l, e: pgd_untargeted(m, x, l, k=20, eps=e, eps_step=e/10),
    "Targeted_PGD": lambda m, x, t, e: pgd_targeted(m, x, t, k=20, eps=e, eps_step=e/10)
}

experiments = [
    {
        "model": MnistCNN, 
        "loader": mnist_test_loader, 
        "dataset_name": "MNIST", 
        "device": device
    },
    {
        "model": cifarCNN, 
        "loader": cifar_test_loader, 
        "dataset_name": "CIFAR10", 
        "device": device
    }
]

for exp in experiments:

    print(f"{exp['dataset_name']} Start")
    model = exp["model"].to(exp["device"])
    model.eval()
    
    for attack_name, attack_func in attacks.items():
        attack_rate(model=model, loader=exp["loader"], device=exp["device"], attack_func=attack_func, attack_name=attack_name, dataset_name=exp["dataset_name"], eps_list=[0.05, 0.1, 0.2, 0.3])

print("fin")