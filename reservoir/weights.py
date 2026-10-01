"""Exact weight parsing.

Weights are represented internally as ``fractions.Fraction`` so that huge
integers and decimal strings are handled without any precision loss, and no
division by a single weight ever happens anywhere in the library (zero-weight
items are simply never selected).
"""
from fractions import Fraction

__all__ = ["parse_weight"]

_NON_FINITE = {
    "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan",
}


def parse_weight(value):
    """Parse ``value`` into a non-negative exact ``Fraction``.

    Accepted inputs:
      * ``int`` (arbitrarily large, e.g. ``10**30``)
      * ``str``: integers, decimals and scientific notation
        (e.g. ``"1000000000000000000000000000000"``, ``"0.0001"``, ``"1.5e-3"``)
        as well as exact fractions (``"2/3"``)
      * ``float``: converted through its exact binary expansion
      * ``Fraction``: used as-is

    Zero is allowed (the item is then never selected).  Negative, NaN and
    infinite weights are rejected.
    """
    if isinstance(value, Fraction):
        w = value
    elif isinstance(value, bool):
        raise TypeError("boolean is not a valid weight")
    elif isinstance(value, int):
        w = Fraction(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("weight must be finite, got %r" % (value,))
        w = Fraction(value)  # exact binary expansion of the float
    elif isinstance(value, str):
        w = _parse_string(value)
    else:
        raise TypeError("unsupported weight type: %r" % (type(value).__name__,))
    if w < 0:
        raise ValueError("negative weight: %r" % (value,))
    return w


def _parse_string(text):
    t = text.strip()
    if not t:
        raise ValueError("empty weight string")
    if t.lower() in _NON_FINITE:
        raise ValueError("weight must be finite: %r" % (text,))
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid weight string: %r" % (text,)) from exc
