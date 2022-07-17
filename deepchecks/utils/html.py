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
import html
import typing as t

from deepchecks.utils.strings import get_random_string

__all__ = ['imagetag', 'linktag', 'details_tag', 'iframe_tag']


def _stringify(
    value: t.Union[str, t.Mapping[str, str], None],
    param_name: str,
    template: str = '{k}: {v};',
) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, t.Mapping):
        attrs = ((k if k != 'clazz' else 'class', v) for k, v in value.items())
        return ''.join(template.format(k=k, v=v) for k, v in attrs)
    elif value is None:
        return ''
    else:
        name = type(value).__name__
        raise TypeError(f'Unsupported "{param_name}" parameter type - {name}')


def imagetag(
    img: bytes,
    prevent_resize: bool = True,
    style: t.Union[str, t.Mapping[str, str], None] = None,
    **attrs
) -> str:
    """Return html image tag with embedded image."""
    style = _stringify(style, 'style')
    attrs = _stringify(attrs, 'attrs', template='{k}="{v}"')

    if prevent_resize is True:
        style = f'{style}; min-width:max-content; min-height:max-content;'

    attrs = f'{attrs} style="{style}"' if style else ''
    png = base64.b64encode(img).decode('ascii')
    return f'<img src="data:image/png;base64,{png}" {attrs}/>'


def linktag(
    text: str,
    href: t.Optional[str] = None,
    is_for_iframe_with_srcdoc: bool = False,
    style: t.Union[str, t.Mapping[str, str], None] = None,
    **attrs
) -> str:
    """Return html <a> tag.

    Parameters
    ----------
    text : str
        link text
    href : str
        link href attribute
    is_for_iframe_with_srcdoc : bool, default False
        anchor links, in order to work within iframe require additional prefix
        'about:srcdoc'. This flag tells function whether to add that prefix to
        the anchor link or not
    style : Union[str, Mapping[str, str], None], default None
        tag style rules
    **attrs :
        other tag attributes

    Returns
    -------
    str
    """
    style = _stringify(style, 'style')
    attrs = _stringify(attrs, 'attrs', template='{k}="{v}"')

    if href:
        if is_for_iframe_with_srcdoc and href.startswith('#'):
            href = f'about:srcdoc{href}'
        attrs = f'href={href} {attrs}'

    if style:
        attrs = f'{attrs} style="{style}"'

    return f'<a {attrs}>{text}</a>'


def details_tag(
    title: str,
    content: str,
    id: t.Optional[str] = None,  # pylint: disable=redefined-builtin
    attrs: t.Union[str, t.Mapping[str, str], None] = None,
    style: t.Union[str, t.Mapping[str, str], None] = None,
    content_attrs: t.Union[str, t.Mapping[str, str], None] = None,
    content_style: t.Union[str, t.Mapping[str, str], None] = None,
) -> str:
    """Return HTML <details> tag."""
    style = _stringify(style, 'style')
    attrs = _stringify(attrs, 'attrs', template='{k}="{v}"')
    content_style = _stringify(content_style, 'content_style')
    content_attrs = _stringify(content_attrs, 'content_attrs', template='{k}="{v}"')

    if id:
        attrs = f'id="{id}" {attrs}'
    if style:
        attrs = f'{attrs} style="{style}"'
    if content_style:
        content_attrs = f'{content_attrs} style="{content_style}"'

    return f"""
        <details {attrs}>
            <summary>{title}</summary>
            <div {content_attrs}>
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
            pass
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


def iframe_tag(
    *,
    title: str,
    id: t.Optional[str] = None,  # pylint: disable=redefined-builtin
    srcdoc: t.Optional[str] = None,
    height: str = '500px',
    width: str = '100%',
    allow: str = 'fullscreen',
    frameborder: str = '0',
    with_fullscreen_btn: bool = True,
    collapsible: bool = False,
    **attributes
) -> str:
    """Return html iframe tag."""
    id = id or f'deepchecks-result-iframe-{get_random_string()}'

    attributes = {
        'id': id,
        'height': height,
        'width': width,
        'allow': allow,
        'frameborder': frameborder,
        'srcdoc': srcdoc,
        **attributes
    }
    attributes = {
        k: v
        for k, v
        in attributes.items()
        if v is not None
    }

    if 'srcdoc' in attributes:
        attributes['srcdoc'] = html.escape(attributes['srcdoc'])

    if 'clazz' in attributes:
        attributes['class'] = attributes.pop('clazz')

    attributes = '\n'.join([
        f'{k}="{v}"'
        for k, v in attributes.items()
    ])

    if with_fullscreen_btn is False:
        if collapsible is False:
            return f'<iframe {attributes}></iframe>'
        else:
            return details_tag(
                title=title,
                content=f'<iframe {attributes}></iframe>',
                attrs='open class="deepchecks-collapsible"',
                content_attrs='class="deepchecks-collapsible-content"',
                content_style='padding: 0!important;'
            )

    onclick = f"document.querySelector('#{id}').requestFullscreen();"
    fullscreen_btn = f'<button class="deepchecks-fullscreen-btn" onclick="{onclick}"></button>'

    if collapsible is False:
        return f'<div style="position:relative;">{fullscreen_btn}<iframe allowfullscreen {attributes}></iframe></div>'
    else:
        return details_tag(
            title=title,
            content=f'{fullscreen_btn}<iframe allowfullscreen {attributes}></iframe>',
            attrs='open class="deepchecks-collapsible"',
            content_attrs='class="deepchecks-collapsible-content"',
            content_style='padding: 0!important;',
        )
