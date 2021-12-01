import typing as t


__all__ = ['Hashable']


@t.runtime_checkable
class Hashable(t.Protocol):
    def __hash__(self) -> int: ... # pylint: disable=invalid-hash-returned
    def __le__(self, value) -> bool: ... # noqa: D105
    def __lt__(self, value) -> bool: ... # noqa: D105
    def __ge__(self, value) -> bool: ... # noqa: D105
    def __gt__(self, value) -> bool: ... # noqa: D105
    def __eq__(self, value) -> bool: ... # noqa: D105
