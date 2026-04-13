"""
Определение архитектуры свёрточной нейросети для оценки шахматных позиций.
Используется как при обучении, так и при инференсе.
"""

import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Остаточный блок с двумя сверточными слоями и skip connection.
    Сохраняет размерность каналов.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)  # skip connection
        return out


class ChessValueNet(nn.Module):
    """
    Свёрточная нейросеть для оценки шахматной позиции.
    Вход: тензор (batch, 12, 8, 8)
    Выход: скаляр (нормализованная оценка в диапазоне [-1, 1])

    Архитектура:
    - Начальный сверточный слой 12 -> 32 (без Residual, т к меняется число каналов)
    - 2 остаточных блока по 32 канала
    - Сверточный слой 32 -> 64 (понижение размерности)
    - 2 остаточных блока по 64 канала
    - MaxPool2d для уменьшения пространства
    - Сверточный слой 64 -> 128
    - 2 остаточных блока по 128 каналов
    - Global Average Pooling вместо Flatten (уменьшает параметры)
    - Полносвязные слои: 128 -> 256 -> 1
    """

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.res_block1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_block2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.pool = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.res_block3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.down1(x)
        x = self.res_block2(x)
        x = self.pool(x)
        x = self.down2(x)
        x = self.res_block3(x)
        x = self.global_pool(x)
        x = self.fc(x)

        return x