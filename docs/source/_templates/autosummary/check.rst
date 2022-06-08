{% extends "autosummary/class.rst" %}

{% block footer %}

{% set splitname = fullname.split('.') %}

{# checktype = distribution|integrity|methodology... #}
{% set checktype = splitname[-3:-1][1] %}

{# submoduletype = tabular|vision... #}
{% set submoduletype = splitname[1] %}

{% set path = getcwd() + '/source/checks/' + submoduletype + '/' + checktype + '/plot_' + to_snake_case(objname).lower() + '.py' %}

{%- if path_exists(path) %}
Examples
~~~~~~~~

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

 .. figure:: /checks_gallery/{{ submoduletype }}/{{ checktype}}/images/thumb/sphx_glr_plot_{{ to_snake_case(objname).lower() }}_thumb.png
     :alt: {{ objname }}

     :ref:`sphx_glr_checks_gallery_{{submoduletype}}_{{ checktype }}_plot_{{ to_snake_case(objname).lower() }}.py`

.. raw:: html

    </div>
    <div style="clear:both"></div>
{%- endif %}


{% endblock %}