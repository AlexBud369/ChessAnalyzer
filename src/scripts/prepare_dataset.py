"""
Функции для преобразования FEN-позиции из CSV в тензоры 8x8x12
и сохранения обучающих/валидационных/тестовых выборок.
Обрабатываются только первые MAX_ROWS строк (позиций).
"""
import pandas as pd
import numpy as np
import chess
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from chess_constants import BOARD_RANKS, BOARD_FILES, PIECE_TO_CHANNEL
from dataset_utils import parse_evaluation

CSV_PATH = "data/raw/chessData.csv"
OUTPUT_DIR = "data/processed"
MAX_ROWS = 2_000_000
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_SEED = 42

def fen_to_tensor(fen):
    """Преобразует FEN в тензор 8 x 8 x 12"""
    board = chess.Board(fen)
    tensor = np.zeros((BOARD_RANKS, BOARD_FILES, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = BOARD_RANKS - 1 - chess.square_rank(square)
        col = chess.square_file(square)
        channel = PIECE_TO_CHANNEL[piece.piece_type]
        if piece.color == chess.WHITE:
            tensor[row, col, channel] = 1
        else:
            tensor[row, col, channel + 6] = 1
    return tensor

def load_data(csv_path, max_rows):
    print(f"Загрузка {csv_path} (первые {max_rows} строк)...")
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Загружено {len(df)} строк.")
    return df

def transform_data(df):
    X_list = []
    y_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Преобразование"):
        fen = row['FEN']
        eval_str = row['Evaluation']
        try:
            tensor = fen_to_tensor(fen)
            score = parse_evaluation(eval_str)
            if score is None:
                continue
            X_list.append(tensor)
            y_list.append(score)
        except Exception as e:
            print(f"Ошибка в строке {idx}: {e}")
            continue

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Успешно преобразовано {len(X)} позиций.")
    return X, y

def split_and_save(X, y, output_dir, test_size, val_size, random_seed):
    X, y = shuffle(X, y, random_state=random_seed)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=random_seed
    )

    np.save(os.path.join(output_dir, 'train_X.npy'), X_train)
    np.save(os.path.join(output_dir, 'train_y.npy'), y_train)
    np.save(os.path.join(output_dir, 'val_X.npy'), X_val)
    np.save(os.path.join(output_dir, 'val_y.npy'), y_val)
    np.save(os.path.join(output_dir, 'test_X.npy'), X_test)
    np.save(os.path.join(output_dir, 'test_y.npy'), y_test)

    return {
        "num_train": len(y_train),
        "num_val": len(y_val),
        "num_test": len(y_test),
        "shape": (BOARD_RANKS, BOARD_FILES, 12),
        "eval_min": float(np.min(y)),
        "eval_max": float(np.max(y)),
        "eval_mean": float(np.mean(y)),
        "eval_std": float(np.std(y)),
    }

def save_metadata(metadata, output_dir):
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(CSV_PATH, MAX_ROWS)
    X, y = transform_data(df)
    metadata = split_and_save(X, y, OUTPUT_DIR, TEST_SIZE, VAL_SIZE, RANDOM_SEED)
    save_metadata(metadata, OUTPUT_DIR)

    print("Статистика оценок в обработанных данных:")
    print(metadata)

if __name__ == "__main__":
    main()