from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from chess_engine.model import build_model_from_config
from chess_engine.moves import MOVE_ENCODER
from data.dataset import ChessChunkDataset, discover_chunks


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    _, indices = logits.topk(k, dim=1)
    correct = (indices == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct


def lr_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    value_coef: float,
) -> dict:
    model.eval()
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    top1 = top5 = 0.0
    n_batches = 0
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for planes, policy, value in loader:
        planes = planes.to(device)
        policy = policy.to(device)
        value = value.to(device).unsqueeze(1)
        pl, vl = model(planes)
        policy_loss_sum += criterion_policy(pl, policy).item()
        value_loss_sum += criterion_value(vl, value).item()
        top1 += top_k_accuracy(pl, policy, 1)
        top5 += top_k_accuracy(pl, policy, 5)
        n_batches += 1

    if n_batches == 0:
        return {}
    return {
        "policy_loss": policy_loss_sum / n_batches,
        "value_loss": value_loss_sum / n_batches,
        "total_loss": (policy_loss_sum + value_coef * value_loss_sum) / n_batches,
        "top1": top1 / n_batches,
        "top5": top5 / n_batches,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler_step,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    value_coef: float,
    grad_clip: float,
    use_amp: bool,
    global_step: int,
) -> tuple[dict, int]:
    model.train()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    stats = {"loss": 0.0, "top1": 0.0, "n": 0}

    for planes, policy, value in tqdm(loader, desc="Train", leave=False):
        planes = planes.to(device, non_blocking=True)
        policy = policy.to(device, non_blocking=True)
        value = value.to(device, non_blocking=True).unsqueeze(1)

        for pg in optimizer.param_groups:
            pg["lr"] = scheduler_step(global_step)
        global_step += 1

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            pl, vl = model(planes)
            loss_p = criterion_policy(pl, policy)
            loss_v = criterion_value(vl, value)
            loss = value_coef * loss_v + loss_p

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        stats["loss"] += loss.item()
        stats["top1"] += top_k_accuracy(pl.detach(), policy, 1)
        stats["n"] += 1

    n = max(1, stats["n"])
    return {"loss": stats["loss"] / n, "top1": stats["top1"] / n}, global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    data_dir = Path(args.data_dir or cfg["data"]["data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tcfg.get("amp", True) and device.type == "cuda"

    print("=== Настройки обучения ===")
    if device.type == "cuda":
        print(f"Устройство: CUDA ({torch.cuda.get_device_name(0)})")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / (1024 ** 3):.1f} ГБ")
    else:
        print("Устройство: CPU (CUDA недоступна)")
    print(f"AMP (смешанная точность): {'вкл' if use_amp else 'выкл'}")
    print(f"Batch size: {tcfg['batch_size']}, эпох: {tcfg['epochs']}")
    print(f"Данные: {data_dir.resolve()}")

    train_started_at = datetime.now()
    training_start_perf = time.perf_counter()
    print(f"Старт обучения: {train_started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    train_chunks = discover_chunks(data_dir, "train")
    val_chunks = discover_chunks(data_dir, "val")
    if not train_chunks:
        raise SystemExit(
            f"Нет чанков в {data_dir / 'train'}. Сначала запустите data.prepare_dataset."
        )

    print("Train-выборка:", flush=True)
    train_ds = ChessChunkDataset(train_chunks)
    if val_chunks:
        print("Val-выборка:", flush=True)
        val_ds = ChessChunkDataset(val_chunks)
    else:
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=tcfg["batch_size"],
            shuffle=False,
            num_workers=tcfg.get("num_workers", 4),
            pin_memory=device.type == "cuda",
        )
        if val_ds
        else None
    )

    model = build_model_from_config(cfg).to(device)
    print(f"Параметров: {model.count_parameters():,}, ходов: {MOVE_ENCODER.num_moves}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr_max"],
        betas=(0.9, 0.999),
        weight_decay=tcfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = Path(cfg["paths"]["best_checkpoint"])
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val_loss = ck.get("best_val_loss", best_val_loss)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup_steps = steps_per_epoch * tcfg.get("warmup_epochs", 1)

    def sched(step: int) -> float:
        return lr_at_step(
            step, total_steps, warmup_steps, tcfg["lr_max"], tcfg["lr_min"]
        )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, tcfg["epochs"]):
        epoch_start = time.perf_counter()
        print(f"\n=== Эпоха {epoch + 1}/{tcfg['epochs']} ===")
        train_stats, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            sched,
            device,
            scaler,
            tcfg["value_coef"],
            tcfg["grad_clip"],
            use_amp,
            global_step,
        )
        epoch_min = (time.perf_counter() - epoch_start) / 60.0
        print(
            f"Train loss={train_stats['loss']:.4f} top1={train_stats['top1']:.3f} "
            f"(время эпохи: {epoch_min:.1f} мин)"
        )

        if val_loader:
            val_stats = validate(
                model, val_loader, device, tcfg["value_coef"]
            )
            print(
                f"Val loss={val_stats['total_loss']:.4f} "
                f"top1={val_stats['top1']:.3f} top5={val_stats['top5']:.3f} "
                f"value_mse={val_stats['value_loss']:.4f}"
            )
            if val_stats["total_loss"] < best_val_loss:
                best_val_loss = val_stats["total_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": cfg,
                        "num_policy_moves": MOVE_ENCODER.num_moves,
                    },
                    best_path,
                )
                print(f"Сохранён лучший чекпоинт: {best_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg,
            },
            ckpt_dir / f"epoch_{epoch:02d}.pt",
        )

    train_finished_at = datetime.now()
    total_min = (time.perf_counter() - training_start_perf) / 60.0
    print("\n=== Обучение завершено ===")
    print(f"Окончание: {train_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Длительность: {total_min:.1f} мин ({total_min / 60:.2f} ч)")
    print(f"Лучший чекпоинт: {best_path.resolve()}")


if __name__ == "__main__":
    main()
