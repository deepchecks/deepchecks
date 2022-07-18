{% block header %}
{{ name | escape | underline }}
{% endblock header%}

{% block content %}

{% set splitname = fullname.split('.') %}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

.. autofunction:: {{ splitname[0] }}.{{ splitname[1] }}.Suite.run

{% endblock content %}

{% block footer %}
{% endblock %}