"""
Скрипт обучения нейросети (Value Network) с возможностью продолжения после прерывания.
Использует подготовленные данные из data/processed/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.chess_value_net import ChessValueNet

DATA_DIR = "data/processed"
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "checkpoint.pt")

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 1
PIN_MEMORY = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}\n")

class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    paths = {
        'train_X': os.path.join(DATA_DIR, "train_X.npy"),
        'train_y': os.path.join(DATA_DIR, "train_y.npy"),
        'val_X': os.path.join(DATA_DIR, "val_X.npy"),
        'val_y': os.path.join(DATA_DIR, "val_y.npy"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}\nСначала запустите prepare_dataset.py")

    print("Загрузка train и val наборов...")
    train_X = np.load(paths['train_X'])
    train_y = np.load(paths['train_y'])
    val_X   = np.load(paths['val_X'])
    val_y   = np.load(paths['val_y'])
    print(f"Загружено: train — {len(train_X)} позиций, val — {len(val_X)} позиций")
    return train_X, train_y, val_X, val_y

def normalize_labels(y_train, y_val):
    max_abs = float(max(abs(y_train.max()), abs(y_train.min())))
    y_train_norm = y_train / max_abs
    y_val_norm   = y_val / max_abs
    return y_train_norm, y_val_norm, max_abs

def save_norm_params(max_abs):
    os.makedirs(MODEL_DIR, exist_ok=True)
    norm_path = os.path.join(MODEL_DIR, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump({"max_abs": max_abs}, f, indent=2)
    print(f"Параметры нормализации сохранены: {norm_path} (max_abs = {max_abs:.4f})")

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Validation", leave=False):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def save_checkpoint(model, optimizer, epoch, best_val_loss, is_best=False):
    """Сохраняет чекпоинт и, если нужно, лучшую модель."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    if is_best:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "chess_nn.pth"))
        print(f"   >>> Лучшая модель сохранена (val_loss = {best_val_loss:.6f})")

def load_checkpoint(model, optimizer):
    """Загружает чекпоинт, для начала обучения с той эпохи, на которой прервался (из-за перегрева)"""
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Загружен чекпоинт: эпоха {epoch}, лучшая val_loss = {best_val_loss:.6f}")
        return epoch, best_val_loss
    else:
        print("Чекпоинт не найден. Начинаем обучение с нуля.")
        return 0, float('inf')

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_X, train_y, val_X, val_y = load_data()
    train_y_norm, val_y_norm, max_abs = normalize_labels(train_y, val_y)
    save_norm_params(max_abs)

    train_dataset = ChessDataset(train_X, train_y_norm)
    val_dataset   = ChessDataset(val_X,   val_y_norm)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ChessValueNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_epoch, best_val_loss = load_checkpoint(model, optimizer)

    print(f"Продолжаем обучение с эпохи {start_epoch+1} до {EPOCHS}\n")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss   = validate(model, val_loader, criterion)
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, epoch, best_val_loss, is_best)

    print(f"\nОбучение завершено! Лучшая val loss: {best_val_loss:.6f}")
    print(f"Финальная модель сохранена в {MODEL_DIR}/chess_nn.pth")


if __name__ == "__main__":
    main()