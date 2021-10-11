from numbers import Number

def threshold(result, min=None, max=None) -> bool:
  if min is None and max is None:
    raise Exception('threshold function must recieve one of "min" or "max" parameters')
  if not isinstance(result, Number):
    raise Exception('threshold works only on numeric results')

  if max and result > max:
    return False
  if min and result < min:
    return False
  return True


