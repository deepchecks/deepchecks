{% block header %}
{{ name | escape | underline }}
{% endblock header%}

{% block content %}
.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :special-members: __init__

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
   :toctree:
{% for item in attributes %}
   {% if item.0 != item.upper().0 %}
   {{ name }}.{{ item }}
   {% endif %}
{% endfor %}
{% endif %}
{% endblock attributes %}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}

.. autosummary::
   :toctree:
{% for item in methods %}
   {% if item != "__init__" %}
   {{ name }}.{{ item }}
   {% endif %}
{%- endfor %}
{% endif %}
{% endblock methods %}

{% endblock content %}

{% block footer %}
{% endblock %}