import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

GRAD_CLIP_NORM = 1.0
CHECKPOINT_KEEP_LAST = 3
CHECKPOINT_INTERVAL = 2000


def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def cleanup_old_checkpoints(model_dir, keep_last=CHECKPOINT_KEEP_LAST):
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
    max_epoch = max(e for e, _ in epoch_files)
    threshold = max_epoch - keep_last
    for epoch, f in epoch_files:
        if epoch < threshold:
            os.remove(f)
            print(f"Removed old checkpoint: {f}")


def find_latest_checkpoint(model_dir):
    best_checkpoint = os.path.join(model_dir, "best_checkpoint.pt")
    if os.path.exists(best_checkpoint):
        print(f"Found best checkpoint: {best_checkpoint}")
        return best_checkpoint

    pattern = os.path.join(model_dir, "checkpoint_epoch*_batch*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    def parse(fname):
        base = os.path.basename(fname)
        parts = base.split('_')
        epoch = int(parts[1].replace('epoch', ''))
        batch = int(parts[2].replace('batch', '').replace('.pt', ''))
        return epoch, batch
    return max(files, key=lambda f: parse(f))


def load_checkpoint(model, optimizer, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    batch = ckpt.get('batch', 0)
    best_val = ckpt.get('best_val_loss', float('inf'))
    return epoch, batch, best_val


def save_checkpoint(model, optimizer, epoch, batch, best_val_loss, model_dir, is_best=False):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"checkpoint_epoch{epoch}_batch{batch}.pt")
    state = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(model_dir, "best_checkpoint.pt")
        torch.save(state, best_path)
        print(f"Best checkpoint saved: {best_path}")

def train_one_batch(model, states, values, move_indices, optimizer, device, value_coef):
    states = states.to(device)
    values = values.to(device)
    move_indices = move_indices.to(device)

    optimizer.zero_grad()
    pred_value, pred_policy = model(states)

    loss_v = nn.MSELoss()(pred_value.view(-1), values)
    loss_p = nn.CrossEntropyLoss()(pred_policy, move_indices)
    loss = value_coef * loss_v + loss_p

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return loss.item(), loss_v.item(), loss_p.item()


def train_epoch(model, dataloader, optimizer, epoch_idx, start_batch, best_val_loss,
                model_dir, device, value_coef):
    model.train()
    total_loss = total_v = total_p = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} training", leave=False)
    for batch_idx, (states, values, moves) in enumerate(pbar):
        if batch_idx < start_batch:
            continue
        loss, loss_v, loss_p = train_one_batch(model, states, values, moves,
                                               optimizer, device, value_coef)
        bs = len(states)
        total_loss += loss * bs
        total_v += loss_v * bs
        total_p += loss_p * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{loss:.4f}", v=loss_v, p=loss_p)

        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch_idx+1, batch_idx+1, best_val_loss, model_dir)
            cleanup_old_checkpoints(model_dir)
    return (total_loss / total_samples,
            total_v / total_samples,
            total_p / total_samples)


def validate(model, dataloader, device, value_coef):
    model.eval()
    total_loss = total_v = total_p = 0.0
    total_samples = 0
    with torch.no_grad():
        for states, values, moves in tqdm(dataloader, desc="Valid", leave=False):
            states = states.to(device)
            values = values.to(device)
            moves = moves.to(device)
            pred_value, pred_policy = model(states)
            loss_v = nn.MSELoss()(pred_value.view(-1), values)
            loss_p = nn.CrossEntropyLoss()(pred_policy, moves)
            loss = value_coef * loss_v + loss_p
            bs = len(states)
            total_loss += loss.item() * bs
            total_v += loss_v.item() * bs
            total_p += loss_p.item() * bs
            total_samples += bs
    return (total_loss / total_samples,
            total_v / total_samples,
            total_p / total_samples)


def print_epoch_summary(epoch, epochs, train_metrics, val_metrics, lr, best_val_loss, prev_val_loss):
    train_t, train_v, train_p = train_metrics
    val_t, val_v, val_p = val_metrics
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{epochs}  |  LR = {lr:.6f}")
    print(f"{'='*60}")
    print(f"TRAIN: total={train_t:.4f} (value={train_v:.4f}, policy={train_p:.4f})")
    print(f"VALID: total={val_t:.4f} (value={val_v:.4f}, policy={val_p:.4f})")
    if prev_val_loss is not None:
        delta = val_t - prev_val_loss
        print(f"Change in val total: {delta:+.4f} {'📈' if delta > 0 else '📉'}")
    gap = val_t - train_t
    if gap > 0.5:
        print(f"Large train/val gap ({gap:.2f}) – possible overfitting")
    elif gap < 0:
        print("Val loss lower than train – good generalization")
    else:
        print(f"Train/val gap = {gap:.2f} (normal)")
    if val_t < best_val_loss:
        print(f"NEW BEST! (was {best_val_loss:.4f})")
    print(f"Current best_val_loss = {min(best_val_loss, val_t):.4f}")
    print(f"{'='*60}\n")