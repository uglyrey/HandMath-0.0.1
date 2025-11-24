import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os

# Устройство
device = torch.device('cpu')

# Класс модели
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 11 * 11)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Функция для обучения
def train_model(epochs=5, batch_size=64, lr=0.001):
    # Трансформы
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Датасеты MNIST
    train_dataset = datasets.MNIST(root='../data/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data/mnist', train=False, download=True, transform=transform)

    # Кастомные символы
    class OffsetDataset(Dataset):
        def __init__(self, dataset, offset):
            self.dataset = dataset
            self.offset = offset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            return img, label + self.offset

    custom_path = '../data/custom_symbols'
    if os.path.exists(custom_path):
        custom_dataset = ImageFolder(root=custom_path, transform=transform)
        custom_dataset = OffsetDataset(custom_dataset, 10)
        train_dataset = ConcatDataset([train_dataset, custom_dataset])

    num_classes = 14
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Модель
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Обучение
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Эпоха {epoch + 1}, потеря: {running_loss / len(train_loader)}")

    # Тест
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Точность: {100 * correct / total}%")

    # Сохранение
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/handmath_model.pth')
    print("Модель сохранена")

# Запуск обучения
if __name__ == "__main__":
    train_model()
