from typing import Optional

def _parse_mate(eval_str: str) -> Optional[float]:
    """Обрабатывает матовые оценки вида #+2, #-3, #2, #+19"""
    rest = eval_str[1:]
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


def _parse_regular_number(eval_str: str) -> Optional[float]:
    """Обрабатывает обычные числа (пешки или сантипешки)"""
    if eval_str.startswith('+'):
        eval_str = eval_str[1:]
    try:
        val = int(eval_str)
        if abs(val) > 300:
            return val / 100.0
        else:
            return float(val)
    except ValueError:
        return None


def parse_evaluation(eval_str: str) -> Optional[float]:
    """Преобразует строку оценки в число пешек (с поддержкой # и больших чисел)"""
    if not isinstance(eval_str, str):
        return None

    eval_str = eval_str.strip()
    if not eval_str:
        return None

    if eval_str.startswith('#'):
        return _parse_mate(eval_str)
    else:
        return _parse_regular_number(eval_str)

