import pytest

from langgraph_openai_serve.placeholder import add, divide, multiply, subtract


def test_add():
    assert add(2, 4) == 6.0
    assert add(2.0, 4.0) == 6.0


def test_subtract():
    assert subtract(4.0, 2.0) == 2.0
    assert subtract(4, 2) == 2.0


def test_multiply():
    assert multiply(2.0, 4.0) == 8.0
    assert multiply(2, 4) == 8.0


def test_divide():
    assert divide(4.0, 2) == 2.0
    assert divide(4, 2) == 2.0

    with pytest.raises(ZeroDivisionError):
        divide(4, 0)
