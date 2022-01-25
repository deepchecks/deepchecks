import typing as t

import pandas as pd
from ipywidgets import HTML
from ipywidgets import Widget

from deepchecks.base import SuiteResult
from deepchecks.base.presentation.abc import Presentation
from deepchecks.utils.strings import get_random_string


__all__ = ["SuiteResultPresentation"]


def get_ipython():
    try:
        from IPython import get_ipython as __get_ipython
        return __get_ipython()
    except ImportError:
        return

class SuiteResultPresentation(Presentation[SuiteResult]):
    
    def __init__(self, value: SuiteResult, **kwargs):
        assert isinstance(value, SuiteResult)
        super().__init__(value, **kwargs)
    
    def as_html(self, *, **kwargs) -> str:
        suite_name = self.value.name
        
        if len(self.value.results) == 0:
            return f"<h1>{suite_name}</h1><p>Suite is empty.</p>"

        suite_id = (
            None
            if 'google.colab' in str(get_ipython())
            else get_random_string()
        )


        



# def display_suite_result(suite_name: str, results: List[Union[CheckResult, CheckFailure]],
#                          html_out=None):  # pragma: no cover
#     """Display results of suite in IPython."""
#     if len(results) == 0:
#         display_html(f"""<h1>{suite_name}</h1><p>Suite is empty.</p>""", raw=True)
#         return
#     if 'google.colab' in str(get_ipython()):
#         unique_id = ''
#     else:
#         unique_id = get_random_string()

#     checks_with_conditions: List[CheckResult] = []
#     checks_wo_conditions_display: List[CheckResult] = []
#     checks_w_condition_display: List[CheckResult] = []
#     others_table = []

#     for result in results:
#         if isinstance(result, CheckResult):
#             if result.have_conditions():
#                 checks_with_conditions.append(result)
#                 if result.have_display():
#                     checks_w_condition_display.append(result)
#             elif result.have_display():
#                 checks_wo_conditions_display.append(result)
#             if not result.have_display():
#                 others_table.append([result.get_header(), 'Nothing found', 2])
#         elif isinstance(result, CheckFailure):
#             error_types = (
#                 errors.DatasetValidationError,
#                 errors.ModelValidationError,
#                 errors.DeepchecksProcessError,
#             )
#             if isinstance(result.exception, error_types):
#                 msg = str(result.exception)
#             else:
#                 msg = result.exception.__class__.__name__ + ': ' + str(result.exception)
#             name = result.header
#             others_table.append([name, msg, 1])
#         else:
#             # Should never reach here!
#             raise errors.DeepchecksValueError(
#                 f"Expecting list of 'CheckResult'|'CheckFailure', but got {type(result)}."
#             )

#     checks_w_condition_display = sorted(checks_w_condition_display, key=lambda it: it.priority)

#     light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 4px;">'

#     icons = """
#     <span style="color: green;display:inline-block">\U00002713</span> /
#     <span style="color: red;display:inline-block">\U00002716</span> /
#     <span style="color: orange;font-weight:bold;display:inline-block">\U00000021</span>
#     """

#     check_names = list(set(it.check.name() for it in results))
#     prologue = (
#         f"The suite is composed of various checks such as: {', '.join(check_names[:3])}, etc..."
#         if len(check_names) > 3
#         else f"The suite is composed of the following checks: {', '.join(check_names)}."
#     )

#     suite_creation_example_link = (
#         'https://docs.deepchecks.com/en/stable/examples/guides/create_a_custom_suite.html'
#         '?utm_source=display_output&utm_medium=referral&utm_campaign=suite_link'
#     )

#     # suite summary
#     summ = f"""
#         <h1 id="summary_{unique_id}">{suite_name}</h1>
#         <p>
#             {prologue}<br>
#             Each check may contain conditions (which will result in pass / fail / warning, represented by {icons})
#             as well as other outputs such as plots or tables.<br>
#             Suites, checks and conditions can all be modified. Read more about
#             <a href={suite_creation_example_link} target="_blank">custom suites</a>.
#         </p>
#         """

#     # can't display plotly widgets in kaggle notebooks
#     if html_out or (is_widgets_enabled() and os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is None):
#         _display_suite_widgets(summ,
#                                unique_id,
#                                checks_with_conditions,
#                                checks_wo_conditions_display,
#                                checks_w_condition_display,
#                                others_table,
#                                light_hr,
#                                html_out)
#     else:
#         _display_suite_no_widgets(summ,
#                                   unique_id,
#                                   checks_with_conditions,
#                                   checks_wo_conditions_display,
#                                   checks_w_condition_display,
#                                   others_table,
#                                   light_hr)
