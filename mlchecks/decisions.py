from numbers import Number
import typing


def threshold(min_limit=None, max_limit=None) -> typing.Callable[[Number], bool]:
    if min_limit is None and max_limit is None:
        raise Exception('threshold function must recieve one of "min" or "max" parameters')

    def decide_threshold(result):
        if not isinstance(result, Number):
            raise Exception('threshold works only on numeric results')
        if max_limit and result > max_limit:
            return False
        if min_limit and result < min_limit:
            return False
        return True

    return decide_threshold
