============================
Viewing Deepchecks Results
============================

This guide demonstrates the different ways you can view a Suite or a Check's results while
working with deepchecks interactively, whether if you're working in a notebook or an IDE, locally or on the cloud.
For details about **saving** or **exporting** your results, 
refer to the :doc:`exporting deepchecks results user guide </user-guide/general/export_save_results>`.

Structure:

- :ref:`showing_results_main_methods`

    - :ref:`show_method`

    - :ref:`show_in_window_method`

- :ref:`alternative_show_options_for_troubleshooting`


.. _showing_results_main_methods:

Main Methods for Showing Results
================================

Deepchecks Check and Suite results can be viewed in a separate window or inline if working within a notebook.

.. _show_method:

``show()`` method
-------------------

This is the main show method. It will show the block inline if possible,
and in a window (using the ``show_in_window()`` functionality) if running from an IDE.

.. code-block:: python

   result.show()

.. _show_in_window_method:

``show_in_window()`` method
---------------------------

This will show the results in a separate window.

It requires installation of the following python libraries: ``pyQt5``, ``pygtwebengine``,
and it may not work on cloud environments such as kaggle and colab.

.. code-block:: python

   result.show_in_window()

.. _alternative_show_options_for_troubleshooting:

Alternative Methods for Showing Results (Troubleshooting)
==============================================================

Some environments may have certain limitations with the display,
that may result with partial output (e.g. without plots) of the main
``show()`` or ``show_in_window()`` methods.

If you experience result display problems, you can consider any of the following options:

- Using one of the alternatives: :ref:`show_in_iframe` or :ref:`show_not_interactive`
  elaborated below.

- | Verifying that the ``jupyter-widgets`` extension is installed and enabled.
  | If you're working on jupyterlab, same for ``jupyterlab-plotly`` extension.
  | Restart your jupter server after these changes so that they will have an effect.

- Exporting the results (e.g. :doc:`saving them to html </user-guide/general/export_save_results>`) 
  and viewing the output file.


.. _show_in_iframe:

``show_in_iframe()`` method
---------------------------------

This is the main alternative in case plots aren't displayed properly.
It embeds the output into an iframe with the html content,
thus does not require any jupyter extensions installed on the environment,
and uses the needed javascript libraries inline. 

.. code-block:: python

   result.show_in_iframe()

.. _show_not_interactive:

``show_not_interactive()`` method
---------------------------------------------

This displays a static version of the suite/check result:
all figures and charts are converted into non-interactive 
(no zoom, no dynamic legend) images.
The result is a less detailed output, that doesnt require any javascript
libraries or jupyter extensions in order to work.

.. code-block:: python

   result.show_not_interactive()