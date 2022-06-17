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
"""Module with display utility functions."""
import base64
import typing as t

__all__ = ['imagetag', 'linktag']


def imagetag(img: bytes) -> str:
    """Return html image tag with embedded image."""
    png = base64.b64encode(img).decode('ascii')
    return f'<img src="data:image/png;base64,{png}"/>'


def linktag(
    text: str,
    style: t.Optional[t.Dict[str, t.Any]] = None,
    is_for_iframe_with_srcdoc : bool = False,
    **kwargs
) -> str:
    """Return html a tag.

    Parameters
    ----------
    style : Optional[Dict[str, Any]], default None
        tag style rules
    is_for_iframe_with_srcdoc : bool, default False
        anchor links, in order to work within iframe require additional prefix
        'about:srcdoc'. This flag tells function whether to add that prefix to
        the anchor link or not
    **kwargs :
        other tag attributes

    Returns
    -------
    str
    """
    if is_for_iframe_with_srcdoc and kwargs.get('href', '').startswith('#'):
        kwargs['href'] = f'about:srcdoc{kwargs["href"]}'

    if style is not None:
        kwargs['style'] = '\n'.join([
            f'{k}: {v};'
            for k, v in style.items()
            if v is not None
        ])

    attrs = ' '.join([f'{k}="{v}"' for k, v in kwargs.items()])
    return f'<a {attrs}>{text}</a>'
