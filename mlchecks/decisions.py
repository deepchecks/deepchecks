from numbers import Number
import typing

__all__ = ['threshold']


def threshold(min=None, max=None) -> typing.Callable[[Number], bool]:
    if min is None and max is None:
      raise Exception('threshold function must recieve one of "min" or "max" parameters')

    def decide_threshold(result):
        if not isinstance(result, Number):
            raise Exception('threshold works only on numeric results')
        if max and result > max:
          return False
        if min and result < min:
          return False
        return True

    return decide_threshold

