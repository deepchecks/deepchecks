import html

from deepchecks.core.display import iframe

import pytest


def test_iframe_default():
    result = iframe()
    assert 'id="deepchecks-result-iframe-' in result
    assert 'height="600px"' in result
    assert 'width="100%"' in result
    assert 'allow="fullscreen"' in result
    assert 'frameborder="0"' in result
    assert "allowfullscreen" in result
    assert "Full Screen" in result


def test_iframe_custom_id():
    result = iframe(id="custom-id")
    assert 'id="custom-id"' in result


def test_iframe_custom_dimensions():
    result = iframe(height="800px", width="50%")
    assert 'height="800px"' in result
    assert 'width="50%"' in result


def test_iframe_no_fullscreen_button():
    result = iframe(with_fullscreen_btn=False)
    assert "<button" not in result


def test_iframe_additional_attributes():
    result = iframe(src="https://example.com")
    assert 'src="https://example.com"' in result


def test_iframe_escapes_srcdoc():
    result = iframe(srcdoc="<script>alert('xss')</script>")
    assert 'srcdoc="&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"' in result


@pytest.mark.parametrize(
    "attr, value",
    [
        ("allow", "microphone"),
        ("frameborder", "1"),
        ("height", "500px"),
        ("width", "90%"),
        ("src", "https://example.com"),
        ("srcdoc", "<div>content</div>"),
    ],
)
def test_iframe_various_attributes(attr, value):
    result = iframe(**{attr: value})
    escaped_value = html.escape(value) if attr == "srcdoc" else value
    assert f'{attr}="{escaped_value}"' in result
