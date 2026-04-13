"""
Создаёт сбалансированный датасет с расширенным тензором (18 каналов).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.position_to_tensor_converter import fen_to_tensor_18ch
from evaluation_parser import parse_evaluation

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed_balanced_18ch"
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_SEED = 42
CLIP_VALUE = 30

LIMITS = {
    "chessData.csv": 1_500_000,
    "random_evals.csv": 400_000,
    "tactic_evals.csv": 100_000,
}
MAX_TOTAL_POSITIONS = 1_000_000

TEST_MODE = False
TEST_LIMIT = 2000


def get_adjusted_limits() -> Tuple[Dict[str, int], int]:
    if TEST_MODE:
        limits = {f: TEST_LIMIT for f in LIMITS}
        max_total = TEST_LIMIT * len(LIMITS)
    else:
        limits = LIMITS.copy()
        max_total = MAX_TOTAL_POSITIONS
    return limits, max_total


def load_dataframe_from_csv(csv_path: str, limit: int) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    if limit <= 0:
        return None
    nrows = None if limit is None else limit
    return pd.read_csv(csv_path, nrows=nrows)


def convert_dataframe_to_tensors(df: pd.DataFrame, desc: str) -> Tuple[List[np.ndarray], List[float]]:
    features = []
    targets = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        fen = row['FEN']
        eval_str = row['Evaluation']
        try:
            tensor = fen_to_tensor_18ch(fen)
            score = parse_evaluation(eval_str)
            if score is None:
                continue
            score = np.clip(score, -CLIP_VALUE, CLIP_VALUE)
            features.append(tensor)
            targets.append(score)
        except Exception as e:
            continue
    return features, targets


def collect_data_until_limit(limits: Dict[str, int], max_total: int) -> Tuple[List[np.ndarray], List[float]]:
    all_features = []
    all_targets = []
    total_collected = 0

    for filename, file_limit in limits.items():
        if file_limit == 0:
            continue
        filepath = os.path.join(RAW_DIR, filename)
        df = load_dataframe_from_csv(filepath, file_limit)
        if df is None or len(df) == 0:
            continue

        features, targets = convert_dataframe_to_tensors(df, f"Обработка {filename}")

        for f, t in zip(features, targets):
            if total_collected >= max_total:
                break
            all_features.append(f)
            all_targets.append(t)
            total_collected += 1

        if total_collected >= max_total:
            break

    return all_features, all_targets


def split_into_train_val_test(X: np.ndarray, y: np.ndarray):
    X, y = shuffle(X, y, random_state=RANDOM_SEED)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_datasets_and_metadata(X_train, X_val, X_test, y_train, y_val, y_test, limits):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'train_X.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'train_y.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'val_X.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'val_y.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'test_X.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'test_y.npy'), y_test)

    all_y = np.concatenate([y_train, y_val, y_test])
    metadata = {
        "num_train": len(y_train),
        "num_val": len(y_val),
        "num_test": len(y_test),
        "eval_min": float(all_y.min()),
        "eval_max": float(all_y.max()),
        "eval_mean": float(all_y.mean()),
        "eval_std": float(all_y.std()),
        "clip_value": CLIP_VALUE,
        "num_channels": 18,
        "source_limits": limits,
        "test_mode": TEST_MODE,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    limits, max_total = get_adjusted_limits()
    print(f"Сбор данных с лимитами: {limits}, общий лимит: {max_total}")

    all_features, all_targets = collect_data_until_limit(limits, max_total)

    print(f"Собрано позиций: {len(all_features)} (цель: {max_total})")
    if not all_features:
        print("Нет данных. Проверьте пути и наличие CSV-файлов в data/raw/")
        return

    X = np.array(all_features)
    y = np.array(all_targets)

    X_train, X_val, X_test, y_train, y_val, y_test = split_into_train_val_test(X, y)

    save_datasets_and_metadata(X_train, X_val, X_test, y_train, y_val, y_test, limits)

    print(f"Сохранено в {OUTPUT_DIR}:")
    print(f"  Train: {len(y_train)} позиций")
    print(f"  Val:   {len(y_val)} позиций")
    print(f"  Test:  {len(y_test)} позиций")


if __name__ == "__main__":
    main()