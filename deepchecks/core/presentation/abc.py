import typing as t
from ipywidgets.widgets import Widget


__all__ = ["Presentation"]


T = t.TypeVar("T")


class Serializer(t.Protocol[T]):
    value: T
    def __init__(self, value: T, **kwargs):
        self.value = value


class HtmlSerializer(Serializer[T]):
    def serialize(self, **kwargs) -> str:
        ...


class JsonSerializer(Serializer[T]):
    def serialize(self, **kwargs) -> t.Dict[t.Any, t.Any]:
        ...


class WidgetSerializer(Serializer[T]):
    def serialize(self, **kwargs) -> Widget:
        ...


class Presentation(t.Protocol[T]):
    value: T

    def __init__(self, value: T, **kwargs):
        self.value = value

    def to_json(self, **kwargs) -> t.Dict[t.Any, t.Any]:
        raise NotImplementedError()

    def to_html(self, **kwargs) -> str:
        raise NotImplementedError()

    def to_widget(self, **kwargs) -> Widget:
        raise NotImplementedError()
