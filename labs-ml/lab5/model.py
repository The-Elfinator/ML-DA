import torch
import torch.nn as nn
from torch import randn


class MyCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, num_conv_layers, kernel_size, dropout_rate):
        super(MyCNN, self).__init__()
        layers = []
        in_channels = input_channels

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, num_filters * (2 ** i), kernel_size=kernel_size, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters * (2 ** i)

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        dummy_input = randn(1, input_channels, 32, 32)
        with torch.no_grad():
            conv_out = self.conv(dummy_input)
        conv_out_size = conv_out.view(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
