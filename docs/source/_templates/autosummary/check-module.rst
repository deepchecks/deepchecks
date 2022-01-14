{% extends "autosummary/module.rst" %}

{% block header %}
{{ name | escape | title | underline }}
{% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :template: autosummary/check.rst
      :toctree:
   {% for item in classes %}
      {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock classes %}
