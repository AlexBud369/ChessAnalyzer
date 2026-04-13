"""
Проверка качества датасета Chess Evaluations.
Загружает CSV, проверяет валидность FEN, парсит оценки, выводит статистику
"""
import pandas as pd
import chess
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_parser import parse_evaluation

def load_data(csv_path, nrows):
    print(f"Загрузка {csv_path}...")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"Загружено строк: {len(df)}")
    return df

def check_columns(df, required_cols):
    if not all(col in df.columns for col in required_cols):
        print("Ошибка: отсутствуют необходимые колонки")
        return False
    return True

def report_missing(df):
    print(f"Пропуски в FEN: {df['FEN'].isna().sum()}")
    print(f"Пропуски в evaluation: {df['Evaluation'].isna().sum()}")

def validate_fens(df, sample_size=100):
    sample_fens = df['FEN'].sample(min(sample_size, len(df)))
    valid_count = 0
    for fen in sample_fens:
        try:
            chess.Board(fen)
            valid_count += 1
        except Exception:
            pass
    print(f"Валидных FEN из {sample_size}: {valid_count}")

def parse_evaluations_and_stats(df):
    df['numeric_eval'] = df['Evaluation'].apply(parse_evaluation)
    valid_evals = df['numeric_eval'].dropna()
    print(f"Корректно распарсенных оценок: {len(valid_evals)} / {len(df)}")
    print("Статистика оценок:")
    print(valid_evals.describe())
    return valid_evals

def plot_histogram(valid_evals, title):
    plt.figure(figsize=(10, 5))
    plt.hist(valid_evals, bins=50, edgecolor='black')
    plt.title(title)
    plt.xlabel('Оценка (пешки)')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    plt.show()

def check_dataset(csv_path, nrows=100000):
    df = load_data(csv_path, nrows)

    required_cols = ['FEN', 'Evaluation']
    if not check_columns(df, required_cols):
        return

    report_missing(df)
    validate_fens(df)
    valid_evals = parse_evaluations_and_stats(df)
    plot_histogram(valid_evals, f'Распределение оценок (первые {nrows} позиций)')

if __name__ == "__main__":
    csv_file = "data/raw/chessData.csv"
    if os.path.exists(csv_file):
        check_dataset(csv_file, nrows=100000)
    else:
        print(f"Файл {csv_file} не найден. Сначала скачайте датасет.")