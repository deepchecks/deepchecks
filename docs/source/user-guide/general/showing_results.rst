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
=================================

Deepchecks Check and Suite results can be viewed in a separate window or inline if working within a notebook.

.. _show_method:

``show()`` method
-------------------

This is the main show method. It will show the check/suite result's HTML output inline, in a notebook cell.

.. code-block:: python

   result.show()

.. _show_in_window_method:


``show_in_window()`` method
---------------------------

This method shows the result in a separate window (using the PyQt5 library).
It starts a PyQt5 application in a separate process therefore this display 
method does not work on cloud platforms like Kaggle or Google Colab.

.. code-block:: python

   result.show_in_window()

Note, PyQt5 library is an optional dependency and it is not installed 
along with Deepchecks automatically, therefore to use ``show_in_window()`` you need 
to install it manually by running the next command:

.. code-block:: sh

   $ pip install PyQt5 pygtwebengine


.. _alternative_show_options_for_troubleshooting:

Alternative Methods for Showing Results (Troubleshooting)
==============================================================

The result's HTML output produced by the ``show()`` method 
requires specific jupyter extensions to display properly.
These extensions may not be installed/enabled by default,
or may have compatibility issues with different environments (e.g. some cloud platforms).

Therefore, if you experience any result display problems, you can consider any of the following 
options to resolve them:

- Using one of the alternatives: :ref:`show_in_iframe` or :ref:`show_not_interactive`,
  elaborated below.

- | Verifying that the ``jupyter-widgets`` extension is installed and enabled.
  | If you're working on jupyterlab, same for ``jupyterlab-plotly`` extension.
  | Restart your jupter server after these changes so that they will have an effect.

- Exporting the results (e.g. :doc:`saving them to html </user-guide/general/export_save_results>`) 
  and viewing the output file.

.. _show_in_iframe:

``show_in_iframe()`` method
---------------------------------

This is the main alternative in case you experience any result display problems.
It uses a different output technique (embedding the result's HTML output into an iframe) and thus doesn't require any jupyter extensions installed.

.. code-block:: python

   result.show_in_iframe()

.. _show_not_interactive:


``show_not_interactive()`` method
---------------------------------------------

This method displays a static version of the suite/check result.
It converts all figures and charts into images (no zoom, no dynamic legend). 
The result is a less detailed output, that doesn't require any javascript
libraries or jupyter extensions in order to work.

.. code-block:: python

   result.show_not_interactive()

To be able to use this method you need to install an optional dependency package called 'kaleido'
that is used to transform figures and charts into images. In order to install it run the command:

.. code-block:: sh
    
   $ pip install kaleido