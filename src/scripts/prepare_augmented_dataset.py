"""
Подготовка датасета с аугментацией (смена цвета)
Размер датасета удваивается за счёт аугментации.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.position_to_tensor_converter import fen_to_tensor_18ch
from evaluation_parser import parse_evaluation

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed_augmented_18ch"
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_SEED = 42
CLIP_VALUE = 30

LIMITS = {
    "chessData.csv": 500_000,
    "random_evals.csv": 150_000,
    "tactic_evals.csv": 50_000,
    "synthetic_material.csv": 200_000,
}
MAX_TOTAL_POSITIONS = 900_000

CHUNK_SIZE = 50000

TEST_MODE = False
TEST_LIMIT = 10000


def get_adjusted_limits():
    if TEST_MODE:
        limits = {fname: TEST_LIMIT for fname in LIMITS}
        max_total = TEST_LIMIT * len(LIMITS)
    else:
        limits = LIMITS.copy()
        max_total = MAX_TOTAL_POSITIONS
    return limits, max_total


def load_dataframe_in_chunks(csv_path: str, limit: int, chunk_size: int = 10000):
    """Генератор, читающий CSV по частям."""
    if not os.path.exists(csv_path):
        return
    if limit <= 0:
        return
    reader = pd.read_csv(csv_path, chunksize=chunk_size, dtype={'Evaluation': str})
    rows_read = 0
    for chunk in reader:
        if rows_read >= limit:
            break
        remaining = limit - rows_read
        yield chunk.iloc[:remaining]
        rows_read += len(chunk)
        if rows_read >= limit:
            break


def augment_by_color_flip(tensor: np.ndarray, score: float):
    """
    Создаёт аугментированную позицию: смена цвета фигур и инверсия оценки.
    tensor: 8x8x18 (каналы: 0..11 фигуры, 12 очередь хода, 13-16 рокировки, 17 взятие на проходе)
    Возвращает новый тензор и новую оценку (-score).
    """
    flipped = tensor.copy()
    # Меняем местами группы каналов
    flipped[:, :, 0:6] = tensor[:, :, 6:12]
    flipped[:, :, 6:12] = tensor[:, :, 0:6]
    # Инвертируем очередь хода
    flipped[:, :, 12] = 1 - tensor[:, :, 12]
    # Рокировки
    flipped[:, :, 13] = tensor[:, :, 15]  # белая короткая <- чёрная короткая
    flipped[:, :, 14] = tensor[:, :, 16]  # белая длинная <- чёрная длинная
    flipped[:, :, 15] = tensor[:, :, 13]  # чёрная короткая <- белая короткая
    flipped[:, :, 16] = tensor[:, :, 14]  # чёрная длинная <- белая длинная
    # Взятие на проходе (канал 17) – оставляем как есть
    new_score = -score
    return flipped, new_score


def process_chunk(df_chunk, desc):
    """Обрабатывает один chunk DataFrame, возвращает списки тензоров и оценок (оригинал + аугмент)."""
    features = []
    targets = []
    for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=desc, leave=False):
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

            aug_tensor, aug_score = augment_by_color_flip(tensor, score)
            features.append(aug_tensor)
            targets.append(aug_score)
        except Exception as e:
            continue
    return features, targets


def collect_data_until_limit(limits, max_total):
    """Собирает данные из всех CSV, записывая промежуточные порции в временные файлы."""
    temp_dir = os.path.join(OUTPUT_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    all_features = []
    all_targets = []
    total_collected = 0
    chunk_idx = 0

    for filename, file_limit in limits.items():
        if file_limit == 0:
            continue
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Предупреждение: файл {filepath} не найден, пропускаем")
            continue

        print(f"Обработка {filename} (лимит {file_limit})")
        for chunk_df in load_dataframe_in_chunks(filepath, file_limit, chunk_size=CHUNK_SIZE):
            features, targets = process_chunk(chunk_df, f"  chunk {chunk_idx}")
            all_features.extend(features)
            all_targets.extend(targets)
            total_collected += len(features)

            if len(all_features) >= CHUNK_SIZE * 2 or total_collected >= max_total * 2:
                temp_X = np.array(all_features, dtype=np.float32)
                temp_y = np.array(all_targets, dtype=np.float32)
                np.save(os.path.join(temp_dir, f"temp_X_{chunk_idx}.npy"), temp_X)
                np.save(os.path.join(temp_dir, f"temp_y_{chunk_idx}.npy"), temp_y)
                print(f"  Сохранён временный чанк {chunk_idx} с {len(all_features)} позициями")
                all_features.clear()
                all_targets.clear()
                chunk_idx += 1
                gc.collect()

            if total_collected >= max_total * 2:
                break
        if total_collected >= max_total * 2:
            break

    if all_features:
        temp_X = np.array(all_features, dtype=np.float32)
        temp_y = np.array(all_targets, dtype=np.float32)
        np.save(os.path.join(temp_dir, f"temp_X_{chunk_idx}.npy"), temp_X)
        np.save(os.path.join(temp_dir, f"temp_y_{chunk_idx}.npy"), temp_y)
        chunk_idx += 1

    print("Объединение временных файлов...")
    all_X = []
    all_y = []
    for i in range(chunk_idx):
        X = np.load(os.path.join(temp_dir, f"temp_X_{i}.npy"))
        y = np.load(os.path.join(temp_dir, f"temp_y_{i}.npy"))
        all_X.append(X)
        all_y.append(y)
    if not all_X:
        return [], []
    X_total = np.concatenate(all_X, axis=0)
    y_total = np.concatenate(all_y, axis=0)

    for i in range(chunk_idx):
        os.remove(os.path.join(temp_dir, f"temp_X_{i}.npy"))
        os.remove(os.path.join(temp_dir, f"temp_y_{i}.npy"))
    os.rmdir(temp_dir)

    return X_total, y_total


def split_and_save(X, y):
    """Перемешивает, разбивает на train/val/test и сохраняет."""
    X, y = shuffle(X, y, random_state=RANDOM_SEED)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=RANDOM_SEED
    )

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
        "augmented": True,
        "test_mode": TEST_MODE,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Сохранено: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")


def main():
    limits, max_total = get_adjusted_limits()
    print(f"Сбор данных с лимитами: {limits}, макс. исходных позиций: {max_total}")
    X, y = collect_data_until_limit(limits, max_total)
    print(f"Собрано позиций (с аугментацией): {len(X)}")
    if len(X) == 0:
        print("Нет данных!")
        return
    split_and_save(X, y)


if __name__ == "__main__":
    main()