"""
Скрипт обучения нейросети (Value Network) с возможностью продолжения после прерывания.
"""

import os
import time
import json
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.chess_value_net import ChessValueNet

DATA_DIR = "data/processed_balanced"
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "checkpoint.pt")
CHECKPOINT_INTERVAL = 2000
CHECKPOINT_KEEP_LAST = 1

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 4
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} итераций\n")


class ChessDataset(Dataset):
    """Датасет для шахматных позиций: тензоры и оценки."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    val_X = np.load(paths['val_X'])
    val_y = np.load(paths['val_y'])
    print(f"Загружено: train — {len(train_X)} позиций, val — {len(val_X)} позиций")
    return train_X, train_y, val_X, val_y


def normalize_labels(y_train: np.ndarray, y_val: np.ndarray, clip_value: int = 30) -> Tuple[np.ndarray, np.ndarray, float]:
    """Обрезает и нормализует целевые значения в диапазон [-1, 1]."""
    y_train = np.clip(y_train, -clip_value, clip_value)
    y_val = np.clip(y_val, -clip_value, clip_value)
    max_abs = float(clip_value)
    y_train_norm = y_train / max_abs
    y_val_norm = y_val / max_abs
    return y_train_norm, y_val_norm, max_abs


def save_norm_params(max_abs: float) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    norm_path = os.path.join(MODEL_DIR, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump({"max_abs": max_abs}, f, indent=2)
    print(f"Параметры нормализации сохранены: {norm_path} (max_abs = {max_abs:.4f})")


def create_dataloaders(train_X: np.ndarray, train_y_norm: np.ndarray,
                       val_X: np.ndarray, val_y_norm: np.ndarray) -> Tuple[DataLoader, DataLoader]:
    """Создаёт DataLoader для train и val."""
    train_dataset = ChessDataset(train_X, train_y_norm)
    val_dataset = ChessDataset(val_X, val_y_norm)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, val_loader


def cleanup_old_checkpoints(keep_last: int = CHECKPOINT_KEEP_LAST) -> None:
    pattern = os.path.join(MODEL_DIR, "checkpoint_epoch*_batch*.pt")
    files = glob.glob(pattern)
    epoch_files = []
    for f in files:
        try:
            epoch = int(f.split('_epoch')[1].split('_')[0])
            epoch_files.append((epoch, f))
        except (ValueError, IndexError):
            continue
    if not epoch_files:
        return
    epoch_files.sort(key=lambda x: x[0])

    max_epoch = max(epoch for epoch, _ in epoch_files)
    threshold = max_epoch - keep_last
    for epoch, f in epoch_files:
        if epoch < threshold:
            os.remove(f)
            print(f"Удалён старый чекпоинт: {f}")


def parse_checkpoint_filename(filepath: str) -> Tuple[int, int]:
    base = os.path.basename(filepath)
    parts = base.split('_')
    epoch = int(parts[1].replace('epoch', ''))
    batch = int(parts[2].replace('batch', '').replace('.pt', ''))
    return epoch, batch


def find_latest_checkpoint() -> Optional[str]:
    pattern = os.path.join(MODEL_DIR, "checkpoint_epoch*_batch*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda f: parse_checkpoint_filename(f))


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    checkpoint_path: str) -> Tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch = checkpoint.get('batch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return epoch, batch, best_val_loss


def load_main_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, float]:
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Загружен основной чекпоинт: эпоха {epoch}, лучшая val_loss = {best_val_loss:.6f}")
        return epoch, best_val_loss
    else:
        print("Чекпоинт не найден. Начинаем обучение с нуля.")
        return 0, float('inf')


def resume_from_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, int, float]:
    """
    Пытается возобновить обучение с последнего промежуточного чекпоинта.
    Если промежуточных нет, загружает основной чекпоинт.
    Возвращает (start_epoch, start_batch, best_val_loss).
    """
    latest = find_latest_checkpoint()
    if latest:
        epoch, batch, best_val_loss = load_checkpoint(model, optimizer, latest)
        print(f"Загружен промежуточный чекпоинт: {latest}")
        print(f"   Эпоха {epoch}, батч {batch}, лучшая val_loss = {best_val_loss:.6f}")
        return epoch, batch, best_val_loss
    else:
        epoch, best_val_loss = load_main_checkpoint(model, optimizer)
        return epoch, 0, best_val_loss


def train_one_batch(model: nn.Module, X_batch: torch.Tensor, y_batch: torch.Tensor,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    """Обучает модель на одном батче, возвращает значение loss."""
    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item() * X_batch.size(0)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, epoch: int, start_batch: int = 0,
                best_val_loss_global: Optional[float] = None) -> float:
    """Обучает одну эпоху, начиная с указанного батча, сохраняет промежуточные чекпоинты"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (X_batch, y_batch) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        if batch_idx < start_batch:
            continue

        batch_loss = train_one_batch(model, X_batch, y_batch, optimizer, criterion)
        total_loss += batch_loss
        total_samples += X_batch.size(0)

        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'batch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss_global if best_val_loss_global is not None else float('inf'),
            }
            checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_epoch{epoch}_batch{batch_idx+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"\n Промежуточный чекпоинт сохранён: {checkpoint_path}")
            cleanup_old_checkpoints()

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    """Вычисляет средний loss на валидационной выборке."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Validation", leave=False):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
    return total_loss / total_samples


def save_main_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, best_val_loss: float, is_best: bool = False) -> None:
    """Сохраняет основной чекпоинт и при необходимости лучшую модель"""
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
        print(f" Лучшая модель сохранена (val_loss = {best_val_loss:.6f})")


def print_training_summary(epoch: int, epochs: int, train_loss: float, val_loss: float) -> None:
    """Выводит информацию о текущей эпохе"""
    print(f"Epoch {epoch}/{epochs}")
    print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")


def main() -> None:
    start_time = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_X, train_y, val_X, val_y = load_data()
    train_y_norm, val_y_norm, max_abs = normalize_labels(train_y, val_y)
    save_norm_params(max_abs)

    train_loader, val_loader = create_dataloaders(train_X, train_y_norm, val_X, val_y_norm)

    model = ChessValueNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_epoch, start_batch, best_val_loss = resume_from_checkpoint(model, optimizer)

    print(f"\nПродолжаем обучение с эпохи {start_epoch + 1}, батча {start_batch + 1} (если применимо) до {EPOCHS}\n")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        current_start_batch = start_batch if epoch == start_epoch + 1 else 0
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch,
                                 start_batch=current_start_batch, best_val_loss_global=best_val_loss)
        val_loss = validate(model, val_loader, criterion)

        print_training_summary(epoch, EPOCHS, train_loss, val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_main_checkpoint(model, optimizer, epoch, best_val_loss, is_best)
        start_batch = 0

    elapsed = time.time() - start_time
    print(f"\nОбщее время: {elapsed:.2f} секунд.")
    print(f"\nОбучение завершено! Лучшая val loss: {best_val_loss:.6f}")
    print(f"Финальная модель сохранена в {MODEL_DIR}/chess_nn.pth")


if __name__ == "__main__":
    main()