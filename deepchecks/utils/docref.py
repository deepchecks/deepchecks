# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Documentation reference utilities."""
import typing as t

from deepchecks import __version__

__all__ = ['doclink']

# TODO:
links = {
    'default': {
        'supported-metrics-by-string': 'https://docs.deepchecks.com/en/stable/user-guide/general/metrics_guide.html#list-of-supported-strings', # pylint: disable=line-too-long  # noqa
        'supported-prediction-format': 'https://docs.deepchecks.com/en/stable/user-guide/tabular/supported_models.html#supported-tasks-and-predictions-format' # pylint: disable=line-too-long  # noqa
    },
    # '0.0.1': {},  # noqa
    # '0.0.2': {},  # noqa
}


def doclink(
        name: str,
        default_link: t.Optional[str] = None,
        template: t.Optional[str] = None,
        package_version: str = __version__
) -> str:
    """Get documentation link.

    Parameters
    ----------
    name: str
        the name of the required link as appears in the links' dictionary.
    default_link: t.Optional[str], default: None
        default like to use if no link corresponding to name was found.
    template: t.Optional[str], default: None
        a string template in which to incorporate the link.
    package_version: str
        which version of the docs to use

    Returns
    -------
    str
        The template text incorporated with the relevant link
    """
    index = (
        links[package_version]
        if package_version in links
        else (links.get('default') or {})
    )

    link = index.get(name) or default_link

    if link is None:
        return ''

    return (
        link
        if template is None
        else template.format(link=link)
    )
