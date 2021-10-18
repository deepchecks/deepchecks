"""String functions."""

__all__ = ['string_baseform']

SPECIAL_CHARS: str = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'


def string_baseform(string: str):
    """Remove special characters from given string.

    Args:
        string (str): string to remove special characters from

    Returns:
        (str): string without special characters
    """
    if not isinstance(string, str):
        return string
    return string.translate(str.maketrans('', '', SPECIAL_CHARS)).lower()
