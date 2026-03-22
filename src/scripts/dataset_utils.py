def parse_evaluation(eval_str):
    """Преобразует строку оценки в число пешек."""
    if not isinstance(eval_str, str):
        return None
    eval_str = eval_str.strip()

    if eval_str.startswith('#'):
        rest = eval_str[1:]
        sign = 1
        if rest.startswith('+'):
            rest = rest[1:]
        elif rest.startswith('-'):
            sign = -1
            rest = rest[1:]
        try:
            int(rest)
            return 10.0 if sign == 1 else -10.0
        except ValueError:
            return None
    try:
        if eval_str.startswith('+'):
            eval_str = eval_str[1:]
        return int(eval_str) / 100.0
    except ValueError:
        return None