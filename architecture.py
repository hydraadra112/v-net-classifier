import torch
from torch import nn
import matplotlib.pyplot as plt
class DR_Classifierv2(nn.Module):
    def __init__(self,  output_shape: int, input_shape: int = 3, hidden_units: int = 64):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.Conv2d(hidden_units * 4, hidden_units * 4, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 8, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units * 8),
            nn.Conv2d(hidden_units  * 8, hidden_units  * 8, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(hidden_units * 8),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 8, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptiveAvgPool(x)
        x = self.classifier(x)
        return x