{% extends "autosummary/class.rst" %}

{% block footer %}

{% set splitname = fullname.split('.') %}

{# checktype = distribution|integrity|methodology... #}
{% set checktype = splitname[-3:-1][1] %}

{# submoduletype = tabular|vision... #}
{% set submoduletype = splitname[1] %}

{% set path = getcwd() + '/source/examples/' + submoduletype + '/checks/' + checktype + '/source/plot_' + to_snake_case(objname).lower() + '.py' %}

{%- if path_exists(path) %}
Examples
~~~~~~~~

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

 .. figure:: /examples/{{ submoduletype }}/checks/{{ checktype}}/examples/images/thumb/sphx_glr_plot_{{ to_snake_case(objname).lower() }}_thumb.png
     :alt: {{ objname }}

     :ref:`sphx_glr_examples_{{submoduletype}}_checks_{{ checktype }}_examples_plot_{{ to_snake_case(objname).lower() }}.py`

.. raw:: html

    </div>
    <div style="clear:both"></div>
{%- endif %}
{% endblock %}