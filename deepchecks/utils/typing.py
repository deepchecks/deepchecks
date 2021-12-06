"""Type definitions."""
from typing_extensions import Protocol, runtime_checkable


__all__ = ['Hashable']


@runtime_checkable
class Hashable(Protocol):
    """Trait for any hashable type that also defines comparison operators."""

    def __hash__(self) -> int: # pylint: disable=invalid-hash-returned, noqa: D105
        ...
    def __le__(self, value) -> bool: # noqa: D105
        ...
    def __lt__(self, value) -> bool: # noqa: D105
        ...
    def __ge__(self, value) -> bool: # noqa: D105
        ...
    def __gt__(self, value) -> bool: # noqa: D105
        ...
    def __eq__(self, value) -> bool: # noqa: D105
        ...
