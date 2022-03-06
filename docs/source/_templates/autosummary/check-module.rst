{% extends "autosummary/module.rst" %}

{% block header %}
{{ name | escape | underline }}
{% endblock %}

{% block members %}
{% if members %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :template: autosummary/check.rst
   :toctree:
{% for item in members %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock members %}
