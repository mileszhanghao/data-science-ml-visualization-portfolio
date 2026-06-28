"""
Portfolio-safe CIFAR-10 model definitions and training utilities.

The original assignment notebook is private. This module keeps the reusable
PyTorch model progression without course instructions, grading text, or output
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


INPUT_SIZE = 3 * 32 * 32
NUM_CLASSES = 10


class LinearBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_SIZE, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.view(-1, INPUT_SIZE))


class OneHiddenLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 300)
        self.fc2 = nn.Linear(300, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(-1, INPUT_SIZE)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TwoHiddenLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(-1, INPUT_SIZE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CompactCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 25, kernel_size=5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(25 * 14 * 14, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv(inputs)))
        x = x.view(-1, 25 * 14 * 14)
        return self.fc(x)


class TwoBlockCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 25, kernel_size=5)
        self.conv2 = nn.Conv2d(25, 50, kernel_size=5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RegularizedTwoBlockCNN(nn.Module):
    def __init__(self, dropout_probability: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 25, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(25)
        self.conv2 = nn.Conv2d(25, 50, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.dropout = nn.Dropout(dropout_probability)
        self.fc2 = nn.Linear(500, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(inputs))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 50 * 5 * 5)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


@dataclass
class EpochMetrics:
    epoch: int
    train_accuracy: float
    validation_accuracy: float


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 1e-3,
) -> list[EpochMetrics]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_accuracy=accuracy(model, train_loader, device),
                validation_accuracy=accuracy(model, validation_loader, device),
            )
        )

    return history
