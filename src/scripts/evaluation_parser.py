from typing import Optional

def _parse_mate(eval_str: str) -> Optional[float]:
    s = eval_str.upper().strip()
    if s.startswith('M'):
        rest = s[1:]
        sign = 1
        if rest.startswith('+'):
            rest = rest[1:]
        elif rest.startswith('-'):
            sign = -1
            rest = rest[1:]
        try:
            int(rest)
            return 30.0 if sign == 1 else -30.0
        except:
            return None
    elif s.startswith('#'):
        rest = s[1:]
        sign = 1
        if rest.startswith('+'):
            rest = rest[1:]
        elif rest.startswith('-'):
            sign = -1
            rest = rest[1:]
        try:
            int(rest)
            return 30.0 if sign == 1 else -30.0
        except:
            return None
    return None

def _parse_centipawns(eval_str: str) -> Optional[float]:
    s = eval_str.strip()
    if s.startswith('+'):
        s = s[1:]
    try:
        centipawns = int(s)
        return centipawns / 100.0
    except:
        return None

def _parse_direct_pawns(eval_str: str) -> Optional[float]:
    try:
        return float(eval_str)
    except:
        return None

def parse_evaluation(eval_str: str) -> Optional[float]:
    if not isinstance(eval_str, str):
        return None
    eval_str = eval_str.strip()
    if not eval_str:
        return None

    if eval_str[0] in ('M', '#'):
        return _parse_mate(eval_str)

    if eval_str[0] in ('+', '-'):
        return _parse_centipawns(eval_str)

    return _parse_direct_pawns(eval_str)