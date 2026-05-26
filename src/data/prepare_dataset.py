from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Optional

import chess
import chess.pgn
import numpy as np
import yaml
from tqdm import tqdm

from chess_engine.encoding import board_to_planes
from chess_engine.moves import MOVE_ENCODER
from data.eval_utils import (
    parse_comment_to_value,
    passes_elo,
    passes_time_control,
)


def iter_training_samples(
    pgn_file,
    min_elo: int,
    min_minutes: int,
    eval_scale: float,
    max_games: Optional[int] = None,
    max_positions: Optional[int] = None,
) -> Iterator[tuple[np.ndarray, int, float]]:
    games = 0
    positions = 0
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        headers = game.headers
        if not passes_elo(headers, min_elo):
            continue
        if not passes_time_control(headers, min_minutes):
            continue

        board = game.board()
        prev_eval_comment: Optional[str] = None
        node = game
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            if prev_eval_comment is not None:
                value = parse_comment_to_value(
                    prev_eval_comment, board.turn, eval_scale
                )
                if value is not None:
                    try:
                        policy_idx = MOVE_ENCODER.encode_move(move)
                    except ValueError:
                        board.push(move)
                        prev_eval_comment = next_node.comment
                        node = next_node
                        continue
                    planes = board_to_planes(board)
                    yield planes, policy_idx, value
                    positions += 1
                    if max_positions and positions >= max_positions:
                        return
            board.push(move)
            prev_eval_comment = next_node.comment
            node = next_node

        games += 1
        if max_games and games >= max_games:
            break


def flush_chunk(
    out_dir: Path,
    chunk_id: int,
    planes: List[np.ndarray],
    policies: List[int],
    values: List[float],
    split: str,
) -> None:
    if not planes:
        return
    arr_p = np.stack(planes).astype(np.float16)
    arr_pol = np.array(policies, dtype=np.int32)
    arr_v = np.array(values, dtype=np.float32)
    path = out_dir / split / f"chunk_{chunk_id:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, planes=arr_p, policy=arr_pol, value=arr_v)
    print(f"Сохранён {path} ({len(planes)} позиций)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка датасета из PGN Lichess")
    parser.add_argument("--pgn", type=str, required=True, help="Путь к .pgn или .pgn.zst")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Остановиться после N позиций (удобно без полного архива)",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    chunk_size = args.chunk_size or dcfg["chunk_size"]
    min_elo = dcfg["min_elo"]
    min_minutes = dcfg["min_time_minutes"]
    eval_scale = dcfg["eval_scale"]
    split_ratio = dcfg["train_val_split"]

    pgn_path = Path(args.pgn)
    out_dir = Path(args.out)

    def sample_stream():
        if pgn_path.suffix == ".zst":
            import io
            import zstandard as zstd

            raw = open(pgn_path, "rb")
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            yield from iter_training_samples(
                text,
                min_elo,
                min_minutes,
                eval_scale,
                args.max_games,
                args.max_positions,
            )
            text.close()
            reader.close()
            raw.close()
        else:
            f = open(pgn_path, encoding="utf-8", errors="replace")
            yield from iter_training_samples(
                f,
                min_elo,
                min_minutes,
                eval_scale,
                args.max_games,
                args.max_positions,
            )
            f.close()

    buffers = {
        "train": {"planes": [], "pol": [], "val": []},
        "val": {"planes": [], "pol": [], "val": []},
    }
    chunk_ids = {"train": 0, "val": 0}
    total = 0

    sample_iter = sample_stream()
    for planes, pol, val in tqdm(sample_iter, desc="Позиции"):
        split = "train" if np.random.random() < split_ratio else "val"
        buf = buffers[split]
        buf["planes"].append(planes)
        buf["pol"].append(pol)
        buf["val"].append(val)
        total += 1
        if len(buf["planes"]) >= chunk_size:
            flush_chunk(
                out_dir,
                chunk_ids[split],
                buf["planes"],
                buf["pol"],
                buf["val"],
                split,
            )
            chunk_ids[split] += 1
            buf["planes"], buf["pol"], buf["val"] = [], [], []

    for split in ("train", "val"):
        buf = buffers[split]
        if buf["planes"]:
            flush_chunk(
                out_dir,
                chunk_ids[split],
                buf["planes"],
                buf["pol"],
                buf["val"],
                split,
            )

    meta = out_dir / "meta.txt"
    meta.write_text(
        f"total_positions={total}\n"
        f"num_policy_classes={MOVE_ENCODER.num_moves}\n"
        f"min_elo={min_elo}\n",
        encoding="utf-8",
    )
    print(f"Готово. Всего позиций: {total}, классов ходов: {MOVE_ENCODER.num_moves}")


if __name__ == "__main__":
    main()
