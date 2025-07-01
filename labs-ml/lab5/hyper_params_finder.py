import json

import optuna
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from lab5.model import MyCNN


def objective(trial):
    num_filters = trial.suggest_int("num_filters", 16, 64, step=16)
    kernel_size = trial.suggest_int("kernel_size", 3, 5)
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 5, 20, step=5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Trial {trial.number}:")
    print(f"  num_filters = {num_filters}, kernel_size = {kernel_size}, "
          f"num_conv_layers = {num_conv_layers}, dropout_rate = {dropout_rate}, "
          f"learning_rate = {learning_rate}, batch_size = {batch_size}",
          f"\nepochs = {num_epochs}")
    model = MyCNN(
        input_channels=3,
        num_classes=10,
        num_filters=num_filters,
        num_conv_layers=num_conv_layers,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Оценка
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Лучшие гиперпараметры:", study.best_params)
print("Лучшая точность:", study.best_value)
with open('hyperparams.txt', 'w') as f:

    f.write("Best Hyperparameters:\n")
    f.write(json.dumps(study.best_params, indent=4))
    f.write("\n\n")

    f.write("Best Accuracy:\n")
    f.write(f"{study.best_value:.4f}\n")
