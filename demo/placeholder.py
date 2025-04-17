"""Placeholder file to provide several sample math calculations.

This module allows the user to make mathematical calculations.
Adapted from: https://realpython.com/python-project-documentation-with-mkdocs/


Examples:
    >>> from langgraph_openai import placeholder
    >>> placeholder.add(2, 4)
    6.0
    >>> placeholder.multiply(2.0, 4.0)
    8.0
    >>> from langgraph_openai.placeholder import divide
    >>> divide(4.0, 2)
    2.0

The module contains the following functions:

- `add(a, b)` - Returns the sum of two numbers.
- `subtract(a, b)` - Returns the difference of two numbers.
- `multiply(a, b)` - Returns the product of two numbers.
- `divide(a, b)` - Returns the quotient of two numbers.
"""


def add(a: float | int, b: float | int) -> float:
    """Compute and return the sum of two numbers.

    Examples:
        >>> add(4.0, 2.0)
        6.0
        >>> add(4, 2)
        6.0

    Args:
        a: A number representing the first addend in the addition.
        b: A number representing the second addend in the addition.

    Returns:
        A number representing the arithmetic sum result of `a` and `b`.
    """
    return float(a + b)


def subtract(a: float | int, b: float | int) -> float:
    """Compute and return the substaction of two numbers.

    Examples:
        >>> subtract(4.0, 2.0)
        2.0
        >>> subtract(4, 2)
        2.0

    Args:
        a: A number representing the first substracter in the substract.
        b:  A number representing the second substracter in the substract.

    Returns:
        A number representing the substract result of `a` and `b`.
    """
    return float(a - b)


def multiply(a: float | int, b: float | int) -> float:
    """Compute and return the multiplication of two numbers.

    Examples:
        >>> multiply(4.0, 2.0)
        8.0
        >>> multiply(4, 2)
        8.0

    Args:
        a: A number representing the first multiplicator in the multiply.
        b: A number representing the second multiplicator in the multiply.

    Returns:
        A number representing the multiplied result of `a` and `b`.
    """
    return float(a * b)


def divide(a: float | int, b: float | int) -> float:
    """Compute and return the division of two numbers.

    Examples:
        >>> divide(4.0, 2.0)
        2.0
        >>> divide(4, 2)
        2.0

    Args:
        a: A number representing the first divider in the divide.
        b: A number representing the second divider in the divide.

    Returns:
        A number representing the division result of `a` and `b`.

    Raises:
        ZeroDivisionError: If `b` is zero.
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")

    return float(a / b)
