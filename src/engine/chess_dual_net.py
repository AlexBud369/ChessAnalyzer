import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ChessDualNet(nn.Module):
    """
    Двуглавая сеть: 10 остаточных блоков (3+3+4).
    Вход: 18 каналов (8 на 8).
    """
    def __init__(self, input_channels=18, dropout=0.1):
        super().__init__()
        # Начальная свёртка
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Блоки на 32 каналах – 3 ResidualBlock
        self.res_block1 = nn.Sequential(
            ResidualBlock(32, dropout),
            ResidualBlock(32, dropout),
            ResidualBlock(32, dropout)
        )

        # Понижение разрешения: 32 в 64
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Блоки на 64 каналах – 3 ResidualBlock
        self.res_block2 = nn.Sequential(
            ResidualBlock(64, dropout),
            ResidualBlock(64, dropout),
            ResidualBlock(64, dropout)
        )

        # MaxPool для уменьшения пространства: 8x8 в 4x4
        self.pool = nn.MaxPool2d(2)

        # Понижение: 64 в 128
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Блоки на 128 каналах – 4 ResidualBlock
        self.res_block3 = nn.Sequential(
            ResidualBlock(128, dropout),
            ResidualBlock(128, dropout),
            ResidualBlock(128, dropout),
            ResidualBlock(128, dropout)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Value head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 4096)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.down1(x)
        x = self.res_block2(x)
        x = self.pool(x)
        x = self.down2(x)
        x = self.res_block3(x)

        v_pooled = self.global_pool(x)
        value = self.value_head(v_pooled)

        policy_logits = self.policy_head(x)

        return value, policy_logits