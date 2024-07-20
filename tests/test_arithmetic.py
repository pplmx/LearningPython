import pytest

from tests.arithmetic import add, divide, multiply, subtract


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 2) == 1
    assert add(-1, -2) == -3
    assert add(1999, 2001) == 4000


def test_subtract():
    assert subtract(1, 2) == -1
    assert subtract(-1, 2) == -3
    assert subtract(-1, -2) == 1
    assert subtract(0, 2) == -2


def test_multiply():
    assert multiply(1, 2) == 2
    assert multiply(-1, 2) == -2
    assert multiply(-1, -2) == 2
    assert multiply(0, 2) == 0


def test_divide():
    assert divide(1, 2) == 0.5


def test_divide_with_0():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


def test_divide_with_floor_division():
    assert divide(1, 2, is_floor_division=True) == 0
