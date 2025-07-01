import numpy as np
from torch import optim
import torch.nn as nn

from Models import Net, generate_data, visualize_regression_multiple, find_best_batch_size, test_and_visualize_models, \
    generate_simple_data

n_samples: int = 10000
n_features: int = 1
n_features_complex = 100
n_targets_complex = 100
n_targets: int = 1
noise: float = 0.8
epochs: int = 30
lr: float = 0.001
batch_sizes: list[int] = [16, 64, 256]
loss: nn.Module = nn.MSELoss()


def test_simple():
    data_simple: list[tuple[np.ndarray, np.ndarray]] = generate_simple_data(n_samples=n_samples, noise=noise)
    best_batch_size: int = find_best_batch_size(data=data_simple, batch_sizes=batch_sizes, input_size=n_features,
                                                output_size=n_targets,
                                                epochs=epochs, lr=lr, loss=loss)
    models: list[Net] = [
        Net(input_size=n_features, output_size=n_targets, is_batch_norm=False, is_dropout=False),
        Net(input_size=n_features, output_size=n_targets, is_batch_norm=True, is_dropout=False),
        Net(input_size=n_features, output_size=n_targets, is_batch_norm=False, is_dropout=True),
        Net(input_size=n_features, output_size=n_targets, is_batch_norm=True, is_dropout=True),
    ]
    test_and_visualize_models(
        models=models,
        data=data_simple,
        batch_size=best_batch_size,
        epochs=epochs,
        lr=lr,
        loss_fn=nn.MSELoss(),
        optimizer_cls=optim.Adam,
        simple_visualization=True
    )


def test_complex():
    data_complex: list[tuple[np.ndarray, np.ndarray]] = generate_data(n_samples=n_samples,
                                                                      n_features=n_features_complex,
                                                                      n_targets=n_targets_complex, noise=noise)
    best_batch_size: int = find_best_batch_size(data=data_complex, batch_sizes=batch_sizes, input_size=n_features_complex,
                                                output_size=n_targets_complex,
                                                epochs=epochs, lr=lr, loss=loss)
    models: list[Net] = [
        Net(input_size=n_features_complex, output_size=n_targets_complex, is_batch_norm=False, is_dropout=False),
        Net(input_size=n_features_complex, output_size=n_targets_complex, is_batch_norm=True, is_dropout=False),
        Net(input_size=n_features_complex, output_size=n_targets_complex, is_batch_norm=False, is_dropout=True),
        Net(input_size=n_features_complex, output_size=n_targets_complex, is_batch_norm=True, is_dropout=True),
    ]
    test_and_visualize_models(
        models=models,
        data=data_complex,
        batch_size=best_batch_size,
        epochs=epochs,
        lr=lr,
        loss_fn=nn.MSELoss(),
        optimizer_cls=optim.Adam,
        simple_visualization=False
    )

test_simple()
test_complex()

#Вывод:
# Лучший размер батча -- 256
# Без дропаута и батч-норм работает лучше, возможно связано с распределением данных +- согласно закону
# Возможно если данные были бы более разбросаны, то лучше показали бы себя дропаут и батч-норм

# Batch size: 16, Average Validation loss across datasets: 0.04570459515104691
# Batch size: 64, Average Validation loss across datasets: 0.043437300715595484
# Batch size: 256, Average Validation loss across datasets: 0.04322804308806857
# Best batch size: 256 with Average Validation loss: 0.04322804308806857
# Testing Model 1: BatchNorm=Off, Dropout=Off
# Dataset 1: Validation Loss = 0.059824938885867596
# Dataset 2: Validation Loss = 0.027346782153472304
# Dataset 3: Validation Loss = 0.0437744646333158
# Testing Model 2: BatchNorm=On, Dropout=Off
# Dataset 1: Validation Loss = 0.05870509287342429
# Dataset 2: Validation Loss = 0.02861533034592867
# Dataset 3: Validation Loss = 0.04328241525217891
# Testing Model 3: BatchNorm=Off, Dropout=On
# Dataset 1: Validation Loss = 0.058198006357997656
# Dataset 2: Validation Loss = 0.028041154611855745
# Dataset 3: Validation Loss = 0.043836903758347034
# Testing Model 4: BatchNorm=On, Dropout=On
# Dataset 1: Validation Loss = 0.0626868112012744
# Dataset 2: Validation Loss = 0.02783992374315858
# Dataset 3: Validation Loss = 0.04503683838993311

