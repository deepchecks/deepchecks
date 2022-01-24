import typing as t
from ipywidgets import Widget


__all__ = ["Presentation"]


T = t.TypeVar("T")


class Presentation(t.Generic[T]):

    def __init__(self, value: T, **kwargs):
        self.value = value
    
    def as_html(self, *, **kwargs) -> str:
        raise NotImplementedError()
    
    def as_json(self, *, **kwargs) -> str:
        raise NotImplementedError()
    
    def as_widget(self, *, **kwargs) -> Widget:
        raise NotImplementedError()