import numpy as np
import torch
from torch.utils.data import Dataset

class MMapChessDataset(Dataset):
    def __init__(self, npz_path, max_samples=None):
        self.npz = np.load(npz_path, mmap_mode='r')
        self.states = self.npz['states']
        self.values = self.npz['values']
        self.moves = self.npz['moves']
        self.len = len(self.states)
        if max_samples and max_samples < self.len:
            self.len = max_samples

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        move = torch.tensor(self.moves[idx], dtype=torch.long)
        return state, value, move