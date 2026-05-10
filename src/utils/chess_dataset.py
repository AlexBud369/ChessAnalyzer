import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ChessAlphaDataset(Dataset):
    def __init__(self, npz_files, max_samples=None):
        self.data = []
        total = 0
        for file in npz_files:
            d = np.load(file)
            n = d['states'].shape[0]
            for i in range(n):
                self.data.append((d['states'][i], d['values'][i], d['moves'][i]))
                total += 1
                if max_samples and total >= max_samples:
                    break
            if max_samples and total >= max_samples:
                break
        print(f"Loaded {len(self.data)} positions from {len(npz_files)} files")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, v, m = self.data[idx]
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(v, dtype=torch.float32),
                torch.tensor(m, dtype=torch.long))


class SingleFileSubset(Dataset):
    def __init__(self, states, values, moves, indices):
        self.states = states
        self.values = values
        self.moves = moves
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (torch.tensor(self.states[idx], dtype=torch.float32),
                torch.tensor(self.values[idx], dtype=torch.float32),
                torch.tensor(self.moves[idx], dtype=torch.long))


def load_data(data_dir, test_mode=False, test_size=5000):
    pattern = os.path.join(data_dir, "*.npz")
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files = [f for f in all_files if "val" in os.path.basename(f)]

    if not train_files or not val_files:
        if len(all_files) == 1:
            print(f"Single file {all_files[0]}, splitting 80/20")
            full = np.load(all_files[0])
            states, values, moves = full['states'], full['values'], full['moves']
            n = len(states)
            n_train = int(0.8 * n)
            perm = np.random.permutation(n)
            train_ds = SingleFileSubset(states, values, moves, perm[:n_train])
            val_ds = SingleFileSubset(states, values, moves, perm[n_train:])
            return train_ds, val_ds
        else:
            split = int(0.8 * len(all_files))
            train_files = all_files[:split]
            val_files = all_files[split:]
            print(f"Train: {len(train_files)} files, Val: {len(val_files)} files")

    if test_mode:
        train_ds = ChessAlphaDataset(train_files, max_samples=test_size)
        val_ds = ChessAlphaDataset(val_files, max_samples=max(1000, test_size//5))
    else:
        train_ds = ChessAlphaDataset(train_files)
        val_ds = ChessAlphaDataset(val_files)
    return train_ds, val_ds