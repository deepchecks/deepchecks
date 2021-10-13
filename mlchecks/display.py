"""Module containing all utility methods for display."""

__all__ = ['format_check_display']

from typing import Callable


def format_check_display(title: str, function: Callable, content: str = None):
    """Format a template for displaying results of checks in HTML.

    Args:
        title (str): Title do display
        function (Callable): Function of the check. Used to take the summary to be displayed
        content (str): Additional content to show. If nothing sent a default "Nothing Found" message is used
    Returns:
        (str): The formatted HTML
    """
    docs = function.__doc__
    summary = docs[:docs.find('\n')]
    content = content or '<p><b>&#x2713;</b> Nothing Found</p>'
    return f"""
    <h4>{title}</h4>
    <p>{summary}</p>
    <div>{content}</div>
    """
