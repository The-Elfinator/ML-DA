from typing import Tuple, List, Optional
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import TensorDataset, random_split, DataLoader

random_seed: int = 42
generator = torch.Generator().manual_seed(random_seed)


class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, is_batch_norm: bool = False, is_dropout: bool = False):
        super(Net, self).__init__()
        self.l1: nn.Linear = nn.Linear(input_size, 512)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(512)
        self.l2: nn.Linear = nn.Linear(512, 128)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.l3: nn.Linear = nn.Linear(128, output_size)
        self.act: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(p=0.5)
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        if self.is_batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        if self.is_dropout:
            x = self.dropout(x)

        x = self.l2(x)
        if self.is_batch_norm:
            x = self.bn2(x)
        x = self.act(x)
        if self.is_dropout:
            x = self.dropout(x)

        x = self.l3(x)
        return x


def train_model(model: Net, train_loader, val_loader, epochs: int = 10, lr: float = 0.001,
                loss_fn: any = nn.MSELoss(), optimizer_cls: any = optim.Adam) -> float:
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred: np.ndarray = model(x_batch)
            loss: nn.Module = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def find_best_batch_size(data: List[Tuple[np.ndarray, np.ndarray]], batch_sizes: List[int], input_size: int,
                         output_size: int, lr: float = 0.001, optimizer_cls: any = optim.Adam,
                         loss: nn.Module = nn.MSELoss(),
                         epochs: int = 10) -> int:
    best_batch_size: Optional[int] = None
    best_val_loss: float = float('inf')

    for batch_size in batch_sizes:
        total_val_loss: int = 0

        for x, y_np in data:
            x: torch.tensor = torch.tensor(x, dtype=torch.float32)
            y: torch.tensor = torch.tensor(y_np, dtype=torch.float32)

            dataset: TensorDataset = TensorDataset(x, y)
            train_size: int = int(0.8 * len(dataset))
            val_size: int = len(dataset) - train_size

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = Net(input_size=input_size, output_size=output_size, is_batch_norm=True, is_dropout=False)

            val_loss: float = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                                          optimizer_cls=optimizer_cls, loss_fn=loss)
            total_val_loss += val_loss

        avg_val_loss = total_val_loss / len(data)
        print(f"Batch size: {batch_size}, Average Validation loss across datasets: {avg_val_loss}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_batch_size = batch_size

    print(f'Best batch size: {best_batch_size} with Average Validation loss: {best_val_loss}')
    return best_batch_size


def generate_data(n_samples: int = 1000, n_features: int = 600, n_targets: int = 600, noise: float = 0.01) -> list[
    tuple[np.ndarray, np.ndarray]]:
    x_sin, y_sin = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=noise)
    y_sin = np.sin(x_sin)

    x_cubic, y_cubic = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=noise)
    y_cubic = np.power(y_cubic, 3)
    y_cubic = y_cubic / (np.abs(x_cubic) + 1)

    x_tanh, y_tanh = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=noise)
    y_tanh = np.tanh(y_tanh)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    y_sin = scaler.fit_transform(y_sin)
    y_cubic = scaler.fit_transform(y_cubic)
    y_tanh = scaler.fit_transform(y_tanh)

    return [(x_sin, y_sin), (x_cubic, y_cubic), (x_tanh, y_tanh)]


def visualize_regression_multiple(datasets_non_linear: List[Tuple[np.ndarray, np.ndarray]],
                                  titles_plt: list[str]) -> None:
    plt.figure(figsize=(18, 6))

    for i, (X, y) in enumerate(datasets_non_linear):
        pca_x = PCA(n_components=2).fit_transform(X)
        pca_y = PCA(n_components=1).fit_transform(y)
        plt.subplot(1, 3, i + 1)
        plt.scatter(pca_x[:, 0], pca_x[:, 1], c=pca_y[:, 0], cmap='viridis', s=50, alpha=1)
        plt.colorbar(label='Output (y)')
        plt.title(titles_plt[i])
        plt.xlabel('PCA Component 1 of X')
        plt.ylabel('PCA Component 2 of X')

    plt.show()


image_ind_comp = 0


def visualize_predictions(model: Net, X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, title: str) -> None:
    pca_x = PCA(n_components=2).fit_transform(X.numpy())
    pca_y_true = PCA(n_components=1).fit_transform(y_true.numpy())
    pca_y_pred = PCA(n_components=1).fit_transform(y_pred.numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=pca_y_true[:, 0], cmap='viridis', label='True values', alpha=0.5, s=50)
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=pca_y_pred[:, 0], cmap='coolwarm', label='Predicted values', alpha=0.5,
                s=50, marker='x')
    plt.colorbar(label='Target Value')
    plt.title(title)
    plt.legend()
    plt.xlabel('PCA Component 1 of X')
    plt.ylabel('PCA Component 2 of X')
    global image_ind_comp
    file_name = f"../lab1/img_complex_{image_ind_comp}"
    image_ind_comp += 1
    plt.savefig(file_name, format='png')


def test_and_visualize_models(
        models: List[Net],
        data: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int,
        epochs: int = 10,
        lr: float = 0.001,
        loss_fn: nn.Module = nn.MSELoss(),
        optimizer_cls: any = optim.Adam,
        simple_visualization: bool = True
) -> None:
    plt.switch_backend('TkAgg')
    for i, model in enumerate(models):
        print(
            f"Testing Model {i + 1}: BatchNorm={'On' if model.is_batch_norm else 'Off'}, Dropout={'On' if model.is_dropout else 'Off'}")

        for j, (X_np, y_np) in enumerate(data):
            x = torch.tensor(X_np, dtype=torch.float32)
            y = torch.tensor(y_np, dtype=torch.float32)

            dataset: TensorDataset = TensorDataset(x, y)
            train_size: int = int(0.8 * len(dataset))
            val_size: int = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                                       generator=generator)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            val_loss = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                loss_fn=loss_fn,
                optimizer_cls=optimizer_cls,
            )
            print(f"Dataset {j + 1}: Validation Loss = {val_loss}")

            model.eval()
            with torch.no_grad():
                all_x, all_y = [], []
                for x_batch, y_batch in val_loader:
                    all_x.append(x_batch)
                    all_y.append(y_batch)
                all_x = torch.cat(all_x, dim=0)
                all_y = torch.cat(all_y, dim=0)
                y_pred = model(all_x).squeeze(-1)
                if simple_visualization:
                    visualize_predictions_simple(all_x, all_y, y_pred, f'Model {i + 1}, Dataset {j + 1}')
                else:
                    visualize_predictions(model, all_x, all_y, y_pred, f'Model {i + 1}, Dataset {j + 1}')


def generate_simple_data(n_samples: int = 1000, noise: float = 0.1) -> list[tuple[np.ndarray, np.ndarray]]:
    x_sin = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_sin = np.sin(x_sin) + np.random.normal(0, noise, size=x_sin.shape)

    x_cubic = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_cubic = np.power(x_cubic, 3) + np.random.normal(0, noise * 10, size=x_cubic.shape)

    x_tanh = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_tanh = np.tanh(x_tanh) + np.random.normal(0, noise, size=x_tanh.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    y_sin = scaler.fit_transform(y_sin)
    y_cubic = scaler.fit_transform(y_cubic)
    y_tanh = scaler.fit_transform(y_tanh)

    return [(x_sin, y_sin), (x_cubic, y_cubic), (x_tanh, y_tanh)]


file_ind = 0


def visualize_predictions_simple(X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
                                 title: str) -> None:
    X_np = X.numpy()
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_np, y_true_np, color='green', alpha=0.5, label='True values', s=50, marker='o')
    plt.scatter(X_np, y_pred_np, color='red', alpha=0.5, label='Predicted values', s=50, marker='x')

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    global file_ind
    file_name = f"../lab1/img_{file_ind}.png"
    file_ind += 1
    plt.savefig(file_name, format='png')
