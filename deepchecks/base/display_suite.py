"""Handle display of suite result."""
# pylint: disable=protected-access
from typing import List, Union

from IPython.core.display import display_html

from deepchecks.base.check import CheckResult, CheckFailure
from deepchecks.base.display_pandas import dataframe_to_html, display_dataframe
from deepchecks.string_utils import split_camel_case
import pandas as pd

__all__ = ['display_suite_result_1', 'display_suite_result_2']


def get_display_exists_icon(exists: bool):
    if exists:
        return '<div style="text-align: center">Yes</div>'
    return '<div style="text-align: center">No</div>'


def display_suite_result_1(name: str, results: List[Union[CheckResult, CheckFailure]]):
    """Display results of suite in IPython."""
    display_html(f'<h1>{name}</h1>', raw=True)
    conditions_table = []
    checks_without_condition_table = []
    errors_table = []

    for result in results:
        if isinstance(result, CheckResult):
            if result.have_conditions():
                for cond_result in result.conditions_results:
                    sort_value = cond_result.get_sort_value()
                    icon = cond_result.get_icon()
                    conditions_table.append([icon, result.header, cond_result.name,
                                             cond_result.details, sort_value])
            else:
                checks_without_condition_table.append([result.header,
                                                       get_display_exists_icon(result.have_display())])
        elif isinstance(result, CheckFailure):
            errors_table.append([str(result.check), str(result.exception), result.exception.__class__.__name__])

    # First print summary
    display_html('<h2>Checks Summary</h2>', raw=True)
    if conditions_table:
        display_html('<h3>With Conditions</h3>'
                     '<p>Checks which have defined conditions, which constitute some limitation on the result'
                     ' of the check. The status defines if the limitation passed or not.</p>', raw=True)
        table = pd.DataFrame(data=conditions_table, columns=['Status', 'Check', 'Condition', 'More Info', 'sort'])
        table.sort_values(by=['sort'], inplace=True)
        table.drop('sort', axis=1, inplace=True)
        display_dataframe(table)
    if checks_without_condition_table:
        display_html('<h3>Without Conditions</h3>'
                     '<p>Checks which does not have defined condition. If an interesting result have been found '
                     'it will be displayed below in the next section of "Display Results"</p>', raw=True)
        table = pd.DataFrame(data=checks_without_condition_table, columns=['Check', 'Has Display?'])
        display_dataframe(table)
    if errors_table:
        display_html('<h3>With Error</h3><p>Checks which raised an error during their run</p>', raw=True)
        table = pd.DataFrame(data=errors_table, columns=['Check', 'Error', 'Type'])
        display_dataframe(table)

    only_check_with_display = [r for r in results
                               if isinstance(r, CheckResult) and r.have_display()]
    # If there are no checks with display doesn't print anything else
    if only_check_with_display:
        checks_not_passed = [r for r in only_check_with_display
                             if r.have_conditions() and not r.passed_conditions()]
        checks_without_condition = [r for r in only_check_with_display
                                    if not r.have_conditions() and r.have_display()]
        checks_passed = [r for r in only_check_with_display
                         if r.have_conditions() and r.passed_conditions() and r.have_display()]

        display_html('<hr><h2>Results Display</h2>', raw=True)
        if checks_not_passed:
            display_html('<h3>Checks with Failed Condition</h3>', raw=True)
            for result in sorted(checks_not_passed, key=lambda x: x.get_conditions_sort_value()):
                result._ipython_display_()
        if checks_without_condition:
            display_html('<h3>Checks without Condition</h3>', raw=True)
            for result in checks_without_condition:
                result._ipython_display_()
        if checks_passed:
            display_html('<h3>Checks with Passed Condition</h3>', raw=True)
            for result in checks_passed:
                result._ipython_display_()


def display_suite_result_2(name: str, results: List[Union[CheckResult, CheckFailure]]):
    """Display results of suite in IPython."""
    conditions_table = []
    display_table = []
    others_table = []
    for result in results:
        if isinstance(result, CheckResult):
            if result.have_conditions():
                for cond_result in result.conditions_results:
                    sort_value = cond_result.get_sort_value()
                    icon = cond_result.get_icon()
                    conditions_table.append([icon, result.header, cond_result.name,
                                             cond_result.details, sort_value])
            if result.have_display():
                display_table.append(result)
            else:
                others_table.append([result.header, 'Nothing found', 2])
        elif isinstance(result, CheckFailure):
            msg = result.exception.__class__.__name__ + ': ' + str(result.exception)
            name = split_camel_case(result.check.__name__)
            others_table.append([name, msg, 1])

    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 1px;">'
    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'
    icons = """
    <span style="color: green;display:inline-block">\U00002713</span> /
    <span style="color: red;display:inline-block">\U00002716</span> /
    <span style="color: orange;font-weight:bold;display:inline-block">\U00000021</span>
    """
    html = f"""
    <h1>{name}</h1>
    <p>The suite is composed of various checks such as: {get_first_3(results)}, etc...<br>
    Each check may contain conditions (which results in {icons}), as well as other outputs such as plots or tables.<br>
    Suites, checks and conditions can all be modified (see tutorial [link]).</p>
    {bold_hr}<h2>Conditions Summary</h2>
    """
    display_html(html, raw=True)
    if conditions_table:
        conditions_table = pd.DataFrame(data=conditions_table,
                                        columns=['Status', 'Check', 'Condition', 'More Info', 'sort'], )
        conditions_table.sort_values(by=['sort'], inplace=True)
        conditions_table.drop('sort', axis=1, inplace=True)
        display_dataframe(conditions_table, hide_index=True)
    else:
        display_html('<p>No conditions defined on checks in the suite.</p>', raw=True)

    display_html(f'{bold_hr}<h2>Additional Outputs</h2>', raw=True)
    if display_table:
        for i, r in enumerate(display_table):
            r._ipython_display_()
            if i < len(display_table) - 1:
                display_html(light_hr, raw=True)
    else:
        display_html('<p>No outputs to show.</p>', raw=True)

    if others_table:
        others_table = pd.DataFrame(data=others_table, columns=['Check', 'Reason', 'sort'])
        others_table.sort_values(by=['sort'], inplace=True)
        others_table.drop('sort', axis=1, inplace=True)
        html = f"""{bold_hr}
        <h2>Other Checks That Weren't Displayed</h2>
        {dataframe_to_html(others_table, hide_index=True)}
        """
        display_html(html, raw=True)


def get_first_3(results: List[Union[CheckResult, CheckFailure]]):
    first_3 = []
    i = 0
    while len(first_3) < 3 and i < len(results):
        curr = results[i]
        curr_name = split_camel_case(curr.check.__name__)
        if curr_name not in first_3:
            first_3.append(curr_name)
        i += 1
    return ', '.join(first_3)
