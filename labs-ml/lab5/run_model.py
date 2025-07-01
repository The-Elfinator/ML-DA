import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from lab5.model import MyCNN

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется устройство CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Используется устройство MPS")
else:
    device = torch.device("cpu")
    print("Используется устройство CPU")

print("Текущее устройство:", device)

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_flatten_size(mdl, input_shape):
    dummy_input = torch.randn(1, *input_shape)
    features = mdl.features(dummy_input)
    return features.view(-1).shape[0]

cifar0_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

batch_size = 128

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=data_transforms,
    download=True
)
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=data_transforms,
    download=True
)



train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader))
print(len(test_loader))

model = MyCNN(input_channels=3, num_classes=10, num_conv_layers=5)
model = model.to(device)

print("Модель на устройстве:", next(model.parameters()).device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

epochs = 30

model.train()

for epoch in range(epochs):
    epoch_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        epoch_loss += loss.item() * images.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_dataset)

    print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}')

model.eval()

correct = 0
total = 0

torch.save(model.state_dict(), "model_weights.pth")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
