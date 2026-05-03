import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.chess_dual_net import ChessDualNet

DEFAULT_DATA_DIR = "src/data/chess_data"
DEFAULT_MODEL_DIR = "src/model1"
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 40
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_VALUE_COEF = 1.0
DEFAULT_NUM_WORKERS = 0
CHECKPOINT_INTERVAL = 2000
CHECKPOINT_KEEP_LAST = 3
GRAD_CLIP_NORM = 1.0


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    return device

class ChessAlphaDataset(Dataset):
    def __init__(self, npz_files, max_samples=None):
        self.data = []
        total_loaded = 0
        for file in npz_files:
            d = np.load(file)
            n = d['states'].shape[0]
            for i in range(n):
                self.data.append((
                    d['states'][i],
                    d['values'][i],
                    d['moves'][i]
                ))
                total_loaded += 1
                if max_samples and total_loaded >= max_samples:
                    break
            if max_samples and total_loaded >= max_samples:
                break
        print(f"Загружено {len(self.data)} позиций из {len(npz_files)} файлов")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, value, move_idx = self.data[idx]
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(value, dtype=torch.float32),
                torch.tensor(move_idx, dtype=torch.long))

def load_data(data_dir: str, test_mode: bool = False, test_size: int = 5000):
    pattern = os.path.join(data_dir, "*.npz")
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"Нет .npz файлов в {data_dir}")

    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files = [f for f in all_files if "val" in os.path.basename(f)]

    if not train_files or not val_files:
        if len(all_files) == 1:
            print(f"Найден один файл: {all_files[0]}. Разбиваем на train/val 80/20.")
            full_data = np.load(all_files[0])
            states = full_data['states']
            values = full_data['values']
            moves = full_data['moves']
            n = len(states)
            n_train = int(0.8 * n)
            indices = np.random.permutation(n)
            train_idx, val_idx = indices[:n_train], indices[n_train:]

            class SubsetDataset(Dataset):
                def __init__(self, states, values, moves, idx_list):
                    self.states = states
                    self.values = values
                    self.moves = moves
                    self.idx_list = idx_list
                def __len__(self):
                    return len(self.idx_list)
                def __getitem__(self, i):
                    idx = self.idx_list[i]
                    return (torch.tensor(self.states[idx], dtype=torch.float32),
                            torch.tensor(self.values[idx], dtype=torch.float32),
                            torch.tensor(self.moves[idx], dtype=torch.long))

            return SubsetDataset(states, values, moves, train_idx), SubsetDataset(states, values, moves, val_idx)
        else:
            split = int(0.8 * len(all_files))
            train_files = all_files[:split]
            val_files = all_files[split:]
            print(f"Используем train: {len(train_files)} файлов, val: {len(val_files)} файлов")

    if test_mode:
        train_dataset = ChessAlphaDataset(train_files, max_samples=test_size)
        val_dataset = ChessAlphaDataset(val_files, max_samples=max(1000, test_size//5))
    else:
        train_dataset = ChessAlphaDataset(train_files)
        val_dataset = ChessAlphaDataset(val_files)
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def cleanup_old_checkpoints(model_dir: str, keep_last: int = CHECKPOINT_KEEP_LAST):
    pattern = os.path.join(model_dir, "checkpoint_epoch*_batch*.pt")
    files = glob.glob(pattern)
    epoch_files = []
    for f in files:
        try:
            epoch = int(f.split('_epoch')[1].split('_')[0])
            epoch_files.append((epoch, f))
        except:
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

def find_latest_checkpoint(model_dir: str) -> Optional[str]:
    pattern = os.path.join(model_dir, "checkpoint_epoch*_batch*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    def parse_epoch_batch(fname):
        base = os.path.basename(fname)
        parts = base.split('_')
        epoch = int(parts[1].replace('epoch', ''))
        batch = int(parts[2].replace('batch', '').replace('.pt', ''))
        return epoch, batch
    return max(files, key=lambda f: parse_epoch_batch(f))

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch = checkpoint.get('batch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    return epoch, batch, best_val_loss

def save_checkpoint(model, optimizer, epoch, batch, best_val_loss, model_dir, is_best=False):
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, f"checkpoint_epoch{epoch}_batch{batch}.pt")
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, ckpt_path)
    if is_best:
        best_path = os.path.join(model_dir, "chess_dual_best.pth")
        torch.save(model.state_dict(), best_path)
        print(f"Лучшая модель сохранена: {best_path}")
    print(f"Чекпоинт сохранён: {ckpt_path}")

def train_one_batch(model, states, values, move_indices, optimizer, device, value_coef):
    states = states.to(device)
    values = values.to(device)
    move_indices = move_indices.to(device)

    optimizer.zero_grad()
    pred_value, pred_policy_logits = model(states)

    loss_v = nn.MSELoss()(pred_value.view(-1), values)

    loss_p = nn.CrossEntropyLoss()(pred_policy_logits, move_indices)
    loss = value_coef * loss_v + loss_p

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return loss.item(), loss_v.item(), loss_p.item()

def train_epoch(model, dataloader, optimizer, epoch, start_batch, best_val_loss, model_dir, device, value_coef):
    model.train()
    total_loss = 0.0
    total_loss_v = 0.0
    total_loss_p = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} training", leave=False)
    for batch_idx, (states, values, move_idxs) in enumerate(pbar):
        if batch_idx < start_batch:
            continue
        loss, loss_v, loss_p = train_one_batch(model, states, values, move_idxs, optimizer, device, value_coef)
        batch_size = len(states)
        total_loss += loss * batch_size
        total_loss_v += loss_v * batch_size
        total_loss_p += loss_p * batch_size
        total_samples += batch_size
        pbar.set_postfix(loss=f"{loss:.4f}", v=loss_v, p=loss_p)

        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch+1, batch_idx+1, best_val_loss, model_dir, is_best=False)
            cleanup_old_checkpoints(model_dir)
    return (total_loss / total_samples,
            total_loss_v / total_samples,
            total_loss_p / total_samples)

def validate(model, dataloader, device, value_coef):
    model.eval()
    total_loss = 0.0
    total_loss_v = 0.0
    total_loss_p = 0.0
    total_samples = 0
    with torch.no_grad():
        for states, values, move_idxs in tqdm(dataloader, desc="Valid", leave=False):
            states = states.to(device)
            values = values.to(device)
            move_idxs = move_idxs.to(device)
            pred_value, pred_policy_logits = model(states)
            loss_v = nn.MSELoss()(pred_value.view(-1), values)
            loss_p = nn.CrossEntropyLoss()(pred_policy_logits, move_idxs)
            loss = value_coef * loss_v + loss_p
            batch_size = len(states)
            total_loss += loss.item() * batch_size
            total_loss_v += loss_v.item() * batch_size
            total_loss_p += loss_p.item() * batch_size
            total_samples += batch_size
    return (total_loss / total_samples,
            total_loss_v / total_samples,
            total_loss_p / total_samples)

def print_epoch_summary(epoch, epochs, train, val, lr, best_val_loss, prev_val_loss=None):
    train_t, train_v, train_p = train
    val_t, val_v, val_p = val

    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{epochs}  |  LR = {lr:.6f}")
    print(f"{'='*60}")
    print(f"TRAIN: total={train_t:.4f}  (value={train_v:.4f}, policy={train_p:.4f})")
    print(f"VALID: total={val_t:.4f}  (value={val_v:.4f}, policy={val_p:.4f})")

    if prev_val_loss is not None:
        delta = val_t - prev_val_loss
        sign = "📈" if delta > 0 else "📉"
        print(f"Change in val total: {delta:+.4f} {sign}")

    gap = val_t - train_t
    if gap > 0.5:
        print(f"⚠ВНИМАНИЕ: большой разрыв train/val ({gap:.2f}) – возможное переобучение")
    elif gap < 0:
        print(f"Отлично: val loss ниже train loss (обобщение хорошее)")
    else:
        print(f"Разрыв train/val = {gap:.2f} (нормально)")

    if val_t < best_val_loss:
        print(f"НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ! (было {best_val_loss:.4f})")

    print(f"Текущий best_val_loss = {min(best_val_loss, val_t):.4f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Обучение двухглавой сети")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--value-coef", type=float, default=DEFAULT_VALUE_COEF,
                        help="Множитель value loss (рекомендуется 1.0 – 2.0)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = get_device()
    print(f"Гиперпараметры:")
    print(f"  batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"  weight_decay={args.weight_decay}, value_coef={args.value_coef}")
    print(f"  градиентный клиппинг: {GRAD_CLIP_NORM}")

    # Загрузка данных
    train_dataset, val_dataset = load_data(args.data_dir, test_mode=args.test, test_size=args.test_size)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args.batch_size, args.num_workers)

    # Модель (dropout = 0.1)
    model = ChessDualNet(input_channels=18, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Восстановление чекпоинта
    latest_ckpt = find_latest_checkpoint(args.model_dir)
    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')
    if latest_ckpt:
        start_epoch, start_batch, best_val_loss = load_checkpoint(model, optimizer, latest_ckpt, device)
        print(f"Возобновляем с {latest_ckpt}: эпоха {start_epoch}, батч {start_batch}, best_val_loss={best_val_loss:.6f}")
        start_epoch += 1
        start_batch = 0
    else:
        print("Обучение с нуля.")

    total_start = time.time()
    prev_val_loss = None

    for epoch in range(start_epoch, args.epochs):
        current_start_batch = start_batch if epoch == start_epoch else 0
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, current_start_batch,
                                    best_val_loss, args.model_dir, device, args.value_coef)
        val_metrics = validate(model, val_loader, device, args.value_coef)

        print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics,
                            scheduler.get_last_lr()[0], best_val_loss, prev_val_loss)

        is_best = val_metrics[0] < best_val_loss
        if is_best:
            best_val_loss = val_metrics[0]

        scheduler.step(val_metrics[0])
        save_checkpoint(model, optimizer, epoch+1, 0, best_val_loss, args.model_dir, is_best=is_best)
        cleanup_old_checkpoints(args.model_dir)

        prev_val_loss = val_metrics[0]

    elapsed = time.time() - total_start
    print(f"\nОбучение завершено за {elapsed:.2f} сек.")
    print(f"Лучшая val loss: {best_val_loss:.6f}")
    print(f"Финальная модель: {args.model_dir}/chess_dual_best.pth")

    print("\n=== Как интерпретировать потери ===")
    print("Идеальные значения для хорошо обученной модели (468k позиций):")
    print("  total loss (val)     ~ 1.0 – 1.5")
    print("  value loss (val)     ~ 0.02 – 0.05 (MSE) → RMSE ~0.14-0.22 → ошибка ~1.5-2.2 пешки")
    print("  policy loss (val)    ~ 1.0 – 1.5 (CrossEntropy) → точность предсказания хода ~30-40%")
    print("  Разрыв train/val     менее 0.3 – признак отсутствия переобучения")
    print("Если val loss не снижается или растёт – увеличьте dropout, weight_decay или уменьшите модель.")

if __name__ == "__main__":
    main()