{% extends "autosummary/class.rst" %}

{% block footer %}

{% set splitname = fullname.split('.') %}

{# checktype = distribution|integrity|methodology... #}
{% set checktype = splitname[-3:-1][1] %}

{# submoduletype = tabular|vision... #}
{% set submoduletype = splitname[1] %}

Example
~~~~~~~

.. nbgallery::
   :name: rst-gallery
   :glob:

   {{objname}} Example Notebook </examples/{{ submoduletype }}/checks/{{ checktype }}/{{ to_snake_case(objname).lower() }}>

{% endblock %}