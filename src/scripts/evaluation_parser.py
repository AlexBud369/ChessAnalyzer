from typing import Optional

def _parse_mate(eval_str: str) -> Optional[float]:
    """
    Обрабатывает матовые оценки вида #+2, #-3,
    Возвращает +-30 условное значение для мата
    """
    rest = eval_str[1:]  # убираем '#'
    sign = 1
    if rest.startswith('+'):
        rest = rest[1:]
    elif rest.startswith('-'):
        sign = -1
        rest = rest[1:]
    try:
        int(rest)
        return 30.0 if sign == 1 else -30.0
    except ValueError:
        return None


def _parse_centipawns(eval_str: str) -> Optional[float]:
    """
    Преобразует сантипешки в пешки (строки по типу +1234, -50)
    """
    if eval_str.startswith('+'):
        eval_str = eval_str[1:]
    try:
        centipawns = int(eval_str)
        return centipawns / 100.0
    except ValueError:
        return None


def parse_evaluation(eval_str: str) -> Optional[float]:
    if not isinstance(eval_str, str):
        return None

    eval_str = eval_str.strip()
    if not eval_str:
        return None

    if eval_str.startswith('#'):
        return _parse_mate(eval_str)
    else:
        return _parse_centipawns(eval_str)