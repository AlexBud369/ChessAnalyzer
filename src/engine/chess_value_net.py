"""
Определение архитектуры свёрточной нейросети для оценки шахматных позиций.
Используется как при обучении, так и при инференсе.
"""

import torch.nn as nn

class ChessValueNet(nn.Module):
    """
    Свёрточная нейросеть для оценки шахматной позиции.
    Вход: тензор (batch, 12, 8, 8)
    Выход: скаляр (нормализованная оценка в диапазоне [-1, 1])
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x