from __future__ import annotations

import re
from typing import Optional, Tuple

import chess


EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
MATE_RE = re.compile(r"#(-?\d+)")


def parse_eval_comment(comment: str) -> Optional[Tuple[float, bool]]:
    if not comment:
        return None
    m = EVAL_RE.search(comment)
    if not m:
        return None
    raw = m.group(1).strip()
    mate = MATE_RE.match(raw)
    if mate:
        moves = int(mate.group(1))
        sign = 1.0 if moves > 0 else -1.0
        n = abs(moves)
        return sign * (1.0 - 0.05 * max(0, n - 1)), True
    try:
        return float(raw), False
    except ValueError:
        return None


def eval_pawns_to_value(eval_white_pawns: float, stm: chess.Color, scale: float = 0.3) -> float:
    import math

    v = math.tanh(eval_white_pawns * scale)
    if stm == chess.BLACK:
        v = -v
    return float(v)


def parse_comment_to_value(
    comment: str, stm: chess.Color, scale: float = 0.3
) -> Optional[float]:
    parsed = parse_eval_comment(comment)
    if parsed is None:
        return None
    ev, is_mate = parsed
    if is_mate:
        return -ev if stm == chess.BLACK else ev
    return eval_pawns_to_value(ev, stm, scale)


def passes_elo(headers: dict, min_elo: int) -> bool:
    try:
        w = int(headers.get("WhiteElo", 0))
        b = int(headers.get("BlackElo", 0))
    except (TypeError, ValueError):
        return False
    return w >= min_elo and b >= min_elo


def passes_time_control(headers: dict, min_minutes: int) -> bool:
    tc = headers.get("TimeControl", "")
    if not tc or tc == "-":
        return True
    if tc.startswith("?"):
        return True
    parts = tc.split("+")
    base = parts[0]
    if base.isdigit():
        seconds = int(base)
        return seconds >= min_minutes * 60
    if "/" in base:
        return True
    return True
