from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessChunkDataset(Dataset):
    def __init__(self, chunk_paths: List[Path], preload: bool = True) -> None:
        self.chunks = sorted(chunk_paths)
        if not self.chunks:
            raise ValueError("Список чанков пуст")

        if preload:
            print(f"Загрузка {len(self.chunks)} чанков в RAM...", flush=True)
            planes_parts: List[np.ndarray] = []
            policy_parts: List[np.ndarray] = []
            value_parts: List[np.ndarray] = []
            for i, p in enumerate(self.chunks, start=1):
                print(f"  чанк {i}/{len(self.chunks)}: {p.name} ...", flush=True)
                with np.load(p) as z:
                    planes_parts.append(z["planes"])
                    policy_parts.append(z["policy"])
                    value_parts.append(z["value"])
            self.planes = np.concatenate(planes_parts, axis=0)
            self.policy = np.concatenate(policy_parts, axis=0)
            self.value = np.concatenate(value_parts, axis=0)
            self._preloaded = True
            n = len(self.planes)
            mb = (self.planes.nbytes + self.policy.nbytes + self.value.nbytes) / (1024**2)
            print(f"  → {n:,} позиций, ~{mb:.0f} МБ в RAM", flush=True)
        else:
            self._preloaded = False
            self._offsets = [0]
            for p in self.chunks:
                with np.load(p) as z:
                    n = z["planes"].shape[0]
                self._offsets.append(self._offsets[-1] + n)

    def __len__(self) -> int:
        if self._preloaded:
            return len(self.planes)
        return self._offsets[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid
        chunk_i = lo
        local = idx - self._offsets[chunk_i]
        return chunk_i, local

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._preloaded:
            return (
                torch.from_numpy(self.planes[idx].astype(np.float32)),
                torch.tensor(int(self.policy[idx]), dtype=torch.long),
                torch.tensor(float(self.value[idx]), dtype=torch.float32),
            )
        chunk_i, local = self._locate(idx)
        with np.load(self.chunks[chunk_i]) as z:
            planes = z["planes"][local].astype(np.float32)
            policy = int(z["policy"][local])
            value = float(z["value"][local])
        return (
            torch.from_numpy(planes),
            torch.tensor(policy, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32),
        )


def discover_chunks(data_dir: Path, split: str) -> List[Path]:
    d = data_dir / split
    if not d.exists():
        return []
    return list(d.glob("chunk_*.npz"))
