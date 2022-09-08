def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int, is_floor_division=False):
    if b == 0:
        raise ZeroDivisionError
    if is_floor_division:
        return a // b
    return a / b
