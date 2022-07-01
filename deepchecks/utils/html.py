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

__all__ = ['imagetag', 'linktag', 'details_tag']


def imagetag(img: bytes) -> str:
    """Return html image tag with embedded image."""
    png = base64.b64encode(img).decode('ascii')
    return f'<img src="data:image/png;base64,{png}"/>'


def linktag(
    text: str,
    style: t.Optional[t.Dict[str, t.Any]] = None,
    is_for_iframe_with_srcdoc: bool = False,
    **kwargs
) -> str:
    """Return html <a> tag.

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


def details_tag(
    title: str, 
    content: str,
    id: t.Optional[str] = None,
    additional_style: t.Union[str, t.Mapping[str, str], None] = None,
    attrs: t.Union[str, t.Mapping[str, str], None] = None
) -> str:
    """Return HTML <details> tag."""
    
    if additional_style is None:
        content_style = ''
    elif isinstance(additional_style, str):
        content_style = additional_style
    elif isinstance(additional_style, t.Mapping):
        content_style = ' '.join(f'{k}: {v};' for k, v in additional_style.items())
    else:
        raise TypeError(f'Unsupported "additional_style" type - {type(additional_style).__name__}')

    if attrs is None:
        attributes = ''
    elif isinstance(attrs, str):
        attributes = attrs
    elif isinstance(attrs, t.Mapping):
        attributes = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    else:
        raise TypeError(f'Unsupported "attrs" type - {type(attrs).__name__}')
    
    if id:
        attributes = f'id="{id}" {attributes}'

    return f"""
        <details {attributes}>
            <summary><strong>{title}</strong></summary>
            <div style="{content_style}">
            {content}
            </div>
        </details>
    """


def tabs_widget(data: t.Mapping[str, t.Union[str, t.List[str]]]) -> str:
    tab_btn_template = '<button data-tab-index="{index}" onclick="deepchecksOpenTab(event)" {attrs}>{title}</button>'
    tab_content_template = '<div class="deepchecks-tab" data-tab-index="{index}" {attrs}>{content}</div>'
    
    buttons = []
    tabs = []

    for i, (k, v) in enumerate(data.items()):
        if isinstance(v, list):
            v = ''.join(v)
        elif isinstance(v, str):
            v = v
        else:
            raise TypeError(f'Unsupported data value type - {type(v).__name__}')
        
        attrs = 'open' if i == 0 else ''
        buttons.append(tab_btn_template.format(index=i, title=k, attrs=attrs))
        tabs.append(tab_content_template.format(index=i, content=v, attrs=attrs))
    
    template = """
    <div class="deepchecks-tabs">
        <div class="deepchecks-tabs-btns">{buttons}</div>
        {tabs}
    </div>
    <script>
        function deepchecksOpenTab(event) {{
            const btn = event.target;
            if (btn.hasAttribute('open') === true)
                return;
            const tabsWidget = btn.closest('div.deepchecks-tabs');
            const tabsBtns = btn.closest('div.deepchecks-tabs-btns');
            const targetIndex = btn.getAttribute('data-tab-index');
            tabsBtns.querySelectorAll('button[open]').forEach(it => it.removeAttribute('open'));
            btn.setAttribute('open', '');
            tabsWidget.querySelectorAll('div.deepchecks-tab[open]').forEach(it => it.removeAttribute('open'));
            tabsWidget.querySelector(`div.deepchecks-tab[data-tab-index="${{targetIndex}}"]`).setAttribute('open', '');
        }}
    </script>
    """

    return template.format(
        buttons=''.join(buttons),
        tabs=''.join(tabs)
    )