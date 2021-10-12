"""Module containing decision functions used to determine if the result of a decidable check is valid or not"""
from numbers import Number
import typing

__all__ = ['threshold']


def threshold(min_threshold: Number = None, max_threshold: Number = None) -> typing.Callable[[Number], bool]:
    """
    Args:
      min_threshold: Minimal allowed value for input number
      max_threshold: Maximal allowed value for input number

    Returns:
      Decision function - A function accepting a number and returning a boolean
    """

    if min_threshold is None and max_threshold is None:
        raise Exception('threshold function must recieve one of "min" or "max" parameters')

    def decide_threshold(result: Number) -> bool:
        if not isinstance(result, Number):
            raise Exception('threshold works only on numeric results')
        if max_threshold and result > max_threshold:
            return False
        if min_threshold and result < min_threshold:
            return False
        return True

    return decide_threshold

