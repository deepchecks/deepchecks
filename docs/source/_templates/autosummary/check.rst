{% extends "autosummary/class.rst" %}

{% block footer %}

{% set splitname = fullname.split('.') %}
{% set checkname = splitname[-3:-1] %}

Example
~~~~~~~

.. nbgallery::
   :name: rst-gallery
   :glob:

   {{objname}} Example Notebook </examples/checks/{{ checkname[1] }}/{{ to_snake_case(objname).lower() }}>

{% endblock %}