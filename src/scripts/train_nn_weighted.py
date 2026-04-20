"""
Скрипт обучения нейросети (Value Network) с взвешенным MSE loss.
"""

import os
import sys
import time
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.chess_value_net import ChessValueNet

DEFAULT_DATA_DIR = "data/processed_augmented_18ch"
DEFAULT_MODEL_DIR = "models"
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 40 #50
DEFAULT_LR = 0.0002
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_CLIP_VALUE = 30
DEFAULT_ALPHA = 0.5  # коэффициент взвешивания loss
CHECKPOINT_INTERVAL = 2000
CHECKPOINT_KEEP_LAST = 1

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    return device

class WeightedChessDataset(Dataset):
    """Датасет с весами для взвешенной MSE."""
    def __init__(self, X: np.ndarray, y: np.ndarray, max_abs: float, alpha: float):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        y_norm = np.clip(y, -max_abs, max_abs) / max_abs
        self.y = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)
        weights = 1.0 + alpha * np.abs(y_norm)
        self.weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

def load_data(data_dir: str, test_mode: bool = False, test_size: int = 1000):
    paths = {
        'train_X': os.path.join(data_dir, "train_X.npy"),
        'train_y': os.path.join(data_dir, "train_y.npy"),
        'val_X': os.path.join(data_dir, "val_X.npy"),
        'val_y': os.path.join(data_dir, "val_y.npy"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}\nСначала запустите подготовку датасета.")

    print("Загрузка train и val наборов...")
    train_X = np.load(paths['train_X'])
    train_y = np.load(paths['train_y'])
    val_X = np.load(paths['val_X'])
    val_y = np.load(paths['val_y'])

    if test_mode:
        train_X = train_X[:test_size]
        train_y = train_y[:test_size]
        val_X = val_X[:test_size]
        val_y = val_y[:test_size]
        print(f"ТЕСТОВЫЙ РЕЖИМ: используем {test_size} позиций для train и val")

    print(f"Загружено: train — {len(train_X)} позиций, val — {len(val_X)} позиций")
    return train_X, train_y, val_X, val_y

def save_norm_params(model_dir: str, max_abs: float, alpha: float):
    os.makedirs(model_dir, exist_ok=True)
    norm_path = os.path.join(model_dir, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump({"max_abs": max_abs, "alpha": alpha}, f, indent=2)
    print(f"Параметры нормализации сохранены: {norm_path} (max_abs={max_abs}, alpha={alpha})")

def create_dataloaders(train_X, train_y, val_X, val_y, max_abs, alpha,
                       batch_size, num_workers=4, pin_memory=True):
    train_dataset = WeightedChessDataset(train_X, train_y, max_abs, alpha)
    val_dataset = WeightedChessDataset(val_X, val_y, max_abs, alpha)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin_memory)
    return train_loader, val_loader

def cleanup_old_checkpoints(model_dir: str, keep_last: int = CHECKPOINT_KEEP_LAST):
    pattern = os.path.join(model_dir, "checkpoint_epoch*_batch*.pt")
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

def find_latest_checkpoint(model_dir: str) -> Optional[str]:
    pattern = os.path.join(model_dir, "checkpoint_epoch*_batch*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda f: parse_checkpoint_filename(f))

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch = checkpoint.get('batch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return epoch, batch, best_val_loss

def load_main_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         checkpoint_file: str, device: torch.device) -> Tuple[int, float]:
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Загружен основной чекпоинт: эпоха {epoch}, лучшая val_loss = {best_val_loss:.6f}")
        return epoch, best_val_loss
    else:
        print("Чекпоинт не найден. Начинаем обучение с нуля.")
        return 0, float('inf')

def resume_from_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                           model_dir: str, checkpoint_file: str, device: torch.device) -> Tuple[int, int, float]:
    latest = find_latest_checkpoint(model_dir)
    if latest:
        epoch, batch, best_val_loss = load_checkpoint(model, optimizer, latest, device)
        print(f"Загружен промежуточный чекпоинт: {latest}")
        print(f"   Эпоха {epoch}, батч {batch}, лучшая val_loss = {best_val_loss:.6f}")
        return epoch, batch, best_val_loss
    else:
        epoch, best_val_loss = load_main_checkpoint(model, optimizer, checkpoint_file, device)
        return epoch, 0, best_val_loss

def train_one_batch(model, X_batch, y_batch, weights_batch, optimizer, device):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    weights_batch = weights_batch.to(device)
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = torch.mean(weights_batch * (outputs - y_batch) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item() * X_batch.size(0)

def train_epoch(model, dataloader, optimizer, epoch, start_batch, best_val_loss_global,
                checkpoint_interval, model_dir, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, (X_batch, y_batch, w_batch) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        if batch_idx < start_batch:
            continue
        batch_loss = train_one_batch(model, X_batch, y_batch, w_batch, optimizer, device)
        total_loss += batch_loss
        total_samples += X_batch.size(0)

        if (batch_idx + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'batch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss_global if best_val_loss_global is not None else float('inf'),
            }
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch{epoch}_batch{batch_idx+1}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"\nПромежуточный чекпоинт сохранён: {ckpt_path}")
            cleanup_old_checkpoints(model_dir)
    return total_loss / total_samples if total_samples > 0 else 0.0

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch, w_batch in tqdm(dataloader, desc="Validation", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            outputs = model(X_batch)
            loss = torch.mean(w_batch * (outputs - y_batch) ** 2)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
    return total_loss / total_samples

def save_main_checkpoint(model, optimizer, epoch, best_val_loss, is_best,
                         model_dir, checkpoint_file):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, checkpoint_file)
    if is_best:
        torch.save(model.state_dict(), os.path.join(model_dir, "chess_nn.pth"))
        print(f"Лучшая модель сохранена (val_loss = {best_val_loss:.6f})")

def main():
    parser = argparse.ArgumentParser(description="Обучение Value Network с взвешенным MSE")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Директория с train_X.npy, train_y.npy, val_X.npy, val_y.npy")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Директория для сохранения моделей и чекпоинтов")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--clip-value", type=float, default=DEFAULT_CLIP_VALUE,
                        help="Максимальная абсолютная оценка в пешках (должна совпадать с подготовкой датасета)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Коэффициент взвешивания loss (w = 1 + alpha * |y_norm|)")
    parser.add_argument("--test", action="store_true",
                        help="Тестовый режим: использовать только 1000 позиций из train и val")
    parser.add_argument("--test-size", type=int, default=1000,
                        help="Количество позиций в тестовом режиме")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_file = os.path.join(args.model_dir, "checkpoint.pt")

    device = get_device()
    print(f"Параметры: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, "
          f"weight_decay={args.weight_decay}, alpha={args.alpha}, clip_value={args.clip_value}")

    train_X, train_y, val_X, val_y = load_data(args.data_dir, test_mode=args.test, test_size=args.test_size)

    save_norm_params(args.model_dir, args.clip_value, args.alpha)

    train_loader, val_loader = create_dataloaders(
        train_X, train_y, val_X, val_y,
        max_abs=args.clip_value, alpha=args.alpha,
        batch_size=args.batch_size
    )

    model = ChessValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    start_epoch, start_batch, best_val_loss = resume_from_checkpoint(
        model, optimizer, args.model_dir, checkpoint_file, device
    )

    print(f"\nПродолжаем обучение с эпохи {start_epoch + 1}, батча {start_batch + 1} (если применимо) до {args.epochs}\n")
    start_time = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        current_start_batch = start_batch if epoch == start_epoch + 1 else 0
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, current_start_batch, best_val_loss,
            CHECKPOINT_INTERVAL, args.model_dir, device
        )
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train loss (weighted MSE): {train_loss:.6f} | Val loss: {val_loss:.6f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        scheduler.step(val_loss)

        save_main_checkpoint(model, optimizer, epoch, best_val_loss, is_best,
                             args.model_dir, checkpoint_file)
        start_batch = 0

    elapsed = time.time() - start_time
    print(f"\nОбщее время: {elapsed:.2f} секунд.")
    print(f"\nОбучение завершено! Лучшая val loss: {best_val_loss:.6f}")
    print(f"Финальная модель сохранена в {args.model_dir}/chess_nn.pth")

if __name__ == "__main__":
    main()