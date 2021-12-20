{% if not obj.display %}
:orphan:

{% endif %}
{% if obj.name.split('.')[-1][0] == '_' %}
{{ obj.name }}
{% elif obj.name.split('.')|length <= 2 %}
{{ (' '.join(obj.name.split('.')[-1].split('_'))|title) }}
{% else %}
{{ obj.name.split('.')[-1] }}
{% endif %}
=========={{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|prepare_docstring|indent(3) }}

{% endif %}

{% block subpackages %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% if visible_subpackages %}

.. toctree::
   :titlesonly:
   :maxdepth: 3

{% for subpackage in visible_subpackages %}
   {{ subpackage.short_name }}/index.rst
{% endfor %}


{% endif %}
{% endblock %}
{% block submodules %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% if visible_submodules %}
Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for submodule in visible_submodules %}
   {{ submodule.short_name }}/index.rst
{% endfor %}


{% endif %}
{% endblock %}
{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}
{{ obj.type|title }} Contents
{{ "-" * obj.type|length }}---------

{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list|sort(attribute="id") %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list|sort(attribute="id") %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list|sort(attribute="id") %}
{% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list|sort(attribute="id") %}
{% if "show-module-summary" in autoapi_options and (visible_classes or visible_functions) %}
{% block classes scoped %}
{% if visible_classes %}
Classes Summary
~~~~~~~~~~~~~~~

.. autoapisummary::

{% for klass in visible_classes %}
   {{ klass.id }}
{% endfor %}


{% endif %}
{% endblock %}

{% block exceptions scoped %}
{% if visible_exceptions %}
Exceptions Summary
~~~~~~~~~~~~~~~~~~

.. autoapisummary::

{% for klass in visible_exceptions %}
   `{{ klass.id }}`
{% endfor %}


{% endif %}
{% endblock %}

{% block functions scoped %}
{% if visible_functions %}
Functions
~~~~~~~~~

.. autoapisummary::
   :nosignatures:

{% for function in visible_functions %}
   {{ function.id }}
{% endfor %}


{% endif %}
{% endblock %}

{% block attributes scoped %}
{% if visible_attributes %}
Attributes Summary
~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

{% for attribute in visible_attributes %}
   {{ attribute.id }}
{% endfor %}


{% endif %}
{% endblock %}
{% endif %}
Contents
~~~~~~~~~~~~~~~~~~~
{% for obj_item in visible_children|sort(attribute="id") %}
{{ obj_item.render()|indent(0) }}
{% endfor %}
{% endif %}
{% endblock %}