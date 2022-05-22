from typing import Sized


def get_condition_passed_message(iter):
    if isinstance(iter, int):
        num_columns = iter
    elif isinstance(iter, Sized):
        num_columns = len(iter)
    else:
        raise TypeError("iter must be an int or a Sized")

    if num_columns == 0:
        return 'No suitable columns to check were found'

    message = f"Passed for {num_columns} suitable column"
    if num_columns > 1:
        message += 's'
    return message
