============================
Viewing Deepchecks Results
============================

This guide demonstrates the different ways you can view a Suite or a Check's results while
working with deepchecks interactively, whether if you're working in a notebook or an IDE, locally or on the cloud.
For details about **saving** or **exporting** your results, 
refer to the :doc:`following user guide</user-guide/exporting_results>`.

**Structure:**

* `Main Methods for Showing Results <#showing_results_main_methods>`__
* `Additional Show Result Options - Troubleshooting <#additional_show_options_for_troubleshooting`__


__showing_results_main_methods

Showing Results - Main Methods
================================

Deepchecks Check and Suite results can be viewed in a separate window or inline if working within a notebook.

``show()`` method
-------------------

This is the main show method. It will show the block inline if possible,
and in a window (using the `show_in_window()` functionality) if running from an IDE.

.. code-block:: python

    result.show()

`show_in_window()`` method
---------------------------

This will show the results in a separate window.
Requires installation of the following python libraries: ``pyQt5``, `pygtwebengine`.
Will notwork on colud environments such as kaggle and colab.

.. code-block:: python

    result.show_in_window()

__additional_show_options_for_troubleshooting:

Troubleshooting Display with Additional Show Options
=======================================================

Some environments may have certain limitations with the display,
that may result with partial output (e.g. without plots) of the 
``show()`` or `show_in_window()` methods.
If you experience result display problems, you can consider any of the following options:

- Using one of the alternative show methods ``show_in_iframe()`` or 
  ``show_not_interactive()`` elaborated below.
- Verifying that jupyter-widgets extension is installed and enabled.
  If working on jupyterlab, same for jupyterlab-plotly extension.
  Restart your jupter server after these changes so that they will have an effect.
- Exporting the results (e.g. saving to html, see `here </user-guide/exporting_results>`) and viewing the result.


Following is an explanation about the additional show methods,
built in case the default `show()`` method doesn't work in your environemnt.

Alternative show method: `show_in_iframe()`
----------------------------------------------

This is the main alternative in case plots aren't displayed properly.
It embeds the output into an iframe with the html content,
thus does not require any jupyter extensions installed on the environment,
and uses the needed javascript libraries inline. 

.. code-block:: python

    result.show_in_iframe()


Alternative show method: `show_not_interactive()`
---------------------------------------------------

This displays a static version of the suite/check result:
all figures and charts are converted into non-interactive 
(no zoom, no dynamic legend) images.
The result is a less detailed output, that doesnt require any javascript
libraries or jupyter extensions to work.

.. code-block:: python

    result.show_not_interactive()