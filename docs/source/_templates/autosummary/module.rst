{% block header %}
{{ name | escape | underline }}
{% endblock header %}

{% block content %}
.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:

   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock attributes %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:

   {% for item in functions %}
      {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock functions %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:

   {% for item in classes %}
      {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock classes %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:

   {% for item in exceptions %}
      {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock exceptions %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
   
{% for item in modules %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock modules %}

{% endblock content %}