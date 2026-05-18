import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.chess_dual_net import ChessDualNet
from utils.chess_dataset import load_data
from utils.training_helpers import (
    create_dataloaders, find_latest_checkpoint, load_checkpoint,
    save_checkpoint, cleanup_old_checkpoints, train_epoch, validate,
    print_epoch_summary
)

DEFAULT_DATA_DIR = "src/data/chess_m"
DEFAULT_MODEL_DIR = "src/model2"
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 15
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_VALUE_COEF = 1.0
DEFAULT_NUM_WORKERS = 0


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--value-coef", type=float, default=DEFAULT_VALUE_COEF)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--pretrained", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = get_device()
    print(f"Hyperparameters: batch={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"  weight_decay={args.weight_decay}, value_coef={args.value_coef}")

    train_ds, val_ds = load_data(args.data_dir, test_mode=args.test, test_size=args.test_size)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, args.batch_size, args.num_workers)

    model = ChessDualNet(input_channels=18, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')

    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict)
        print("Pretrained loaded. Optimizer and scheduler are reset.")
    else:
        latest = find_latest_checkpoint(args.model_dir)
        if latest:
            start_epoch, start_batch, best_val_loss = load_checkpoint(model, optimizer, latest, device)
            print(f"Resuming from {latest}: epoch {start_epoch}, batch {start_batch}, best_val={best_val_loss:.6f}")
            start_epoch += 1
            start_batch = 0
        else:
            print("Training from scratch.")

    total_start = time.time()
    prev_val = None

    for epoch in range(start_epoch, args.epochs):
        cur_start_batch = start_batch if epoch == start_epoch else 0
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, cur_start_batch,
                                    best_val_loss, args.model_dir, device, args.value_coef)
        val_metrics = validate(model, val_loader, device, args.value_coef)

        print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics,
                            scheduler.get_last_lr()[0], best_val_loss, prev_val)

        if val_metrics[0] < best_val_loss:
            best_val_loss = val_metrics[0]
            is_best = True
        else:
            is_best = False

        scheduler.step(val_metrics[0])
        save_checkpoint(model, optimizer, epoch+1, 0, best_val_loss, args.model_dir, is_best=is_best)
        cleanup_old_checkpoints(args.model_dir)

        prev_val = val_metrics[0]

    elapsed = time.time() - total_start
    print(f"\nTraining finished in {elapsed:.2f} sec.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model: {args.model_dir}/chess_dual_best.pth")


if __name__ == "__main__":
    main()