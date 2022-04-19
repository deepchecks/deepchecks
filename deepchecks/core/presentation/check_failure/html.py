from deepchecks.utils.strings import get_docs_summary
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import HtmlSerializer


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(HtmlSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value
    
    def serialize(self, **kwargs) -> str:
        return ''.join([
            self.prepare_header(),
            self.prepare_summary(),
            self.prepare_error_message()
        ])

    def prepare_header(self) -> str:
        return f'<h4>{self.value.header}</h4>'
    
    def prepare_summary(self) -> str:
        return (
            f'<p>{get_docs_summary(self.value)}</p>' 
            if hasattr(type(self.value), '__doc__')
            else ''
        )
    
    def prepare_error_message(self) -> str:
        return f'<p style="color:red"> {self.value.exception}</p>'