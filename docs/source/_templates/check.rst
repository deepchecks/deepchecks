{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

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
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {% if item.0 != item.upper().0 %}
      {{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}


{% set splitname = fullname.split('.') %}
{% set checkname = splitname[-3:-1] %}

Example
~~~~~~~

.. toctree::
   :maxdepth: 1
   :titlesonly:

   /examples/checks/{{ checkname[0] }}/{{ checkname[1] }}