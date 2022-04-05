{% extends "autosummary/class.rst" %}

{% block footer %}

{% set splitname = fullname.split('.') %}

{# checktype = distribution|integrity|methodology... #}
{% set checktype = splitname[-3:-1][1] %}

{# submoduletype = tabular|vision... #}
{% set submoduletype = splitname[1] %}

Example {{ objname }}
~~~~~~~~~~~~~~~~~~~~~

.. nbgallery::
   :name: rst-gallery
   :glob:

   {{objname}} Example Notebook </examples/{{ submoduletype }}/checks/{{ checktype }}/examples/{{ objname.lower() }}>

{% endblock %}