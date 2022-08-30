=================================
Exporting Deepchecks' Results
=================================

This guide will help you understand how to export deepchecks results, 
which enables saving them for later viewing, sharing the analysis results as a report,
sending to different tools, or for further processing steps 
(e.g. in ci/cd for deciding whether a new model is good enough for deployment).

We will discuss the two main export options, that are relevant for saving Check or Suite results:

- :ref:`Saving an HTML File <export_save_results__save_result_to_html>` - saving the interactive report as a file
- :ref:`Exporting to JSON <export_save_results__export_to_json>` - exporting to a serializeable object that can
  later be used in other Python scopes, and enables processing the checks' output with code.

For sending the results to other tools (e.g. experiment trackers),
you can either use one of the above send the desires results or report, or check out
our :ref:`integrations examples section <user_guide__integrations>` in the user guide.


.. _export_save_results__save_result_to_html:

Save Result as an HTML Report: ``save_as_html()``
====================================================

This method saves the Check or Suite's output as an HTML file.
It expects an output filename or a full file path to the report's destination folder.
If no name given, the default filename will be 'output.html'.


Save as HTML
--------------

.. code-block:: python

   result.save_as_html('my_results.html')
   

For the method's API reference, check out: :meth:`deepchecks.core.SuiteResult.save_as_html` or 
:meth:`deepchecks.core.CheckResult.save_as_html` for saving a Suite or a Check's result, respectively.

.. _export_save_results__export_to_json:

Export Results to JSON: ``to_json()``
============================================

This method takes the check outputs and serializes it to a JSON string format.

The format of the output json includes all info required to reconstruct
the run and its output, including the check's name, parameters, a short description of its purpose,
its result value, and a json of the releant data for displaying its outputs.

If serializing a SuiteResult, the output is an array of the CheckResults JSON's, along with the suite's name.


Serialize Output
-----------------

Save Check/Suite result as a JSON formatted string:

.. code-block:: python

   serialized_output = result.to_json()


Inspect the JSON Structure
--------------------------

Parse resulting json string as dict and observe the JSON's structure:

.. code-block:: python

   import json

   json_output = json.loads(serialized_output)
   
   # to see output keys:
   json_output.keys()
   
   # to see all JSON:
   json_output


Check Result
~~~~~~~~~~~~~~

See docstring in code or in method's API reference: :meth:`deepchecks.core.CheckResult.to_json`.

.. code-block:: python

   from deepchecks.core import CheckResult

   help(CheckResult.to_json)
   
Inspect the check results output keys and content to see how the checks name, parameters, etc. are saved.



Suite Result
~~~~~~~~~~~~~~~

See docstring in code or in the method's API reference: :meth:`deepchecks.core.SuiteResult.to_json`.

A Suite's output JSON is an array of all of the Check's output JSON's inside the ``results`` key, along with the suite's name.

.. code-block:: python

   from deepchecks.core import SuiteResult

   help(SuiteResult.to_json)


See suite result json:

.. code-block:: python

   # save as json string
   suite_json_ = suite_result.to_json()
   # load json to dict from string
   json_output = json.loads(suite_json)

   # Suite name
   json_output['name']

   # Results is an array of CheckResult JSON's, let's see how many checks ran in suite:
   len(json_output['results'])

   # Let's observe a specific check's output, which is similar to a single check's json format`Check output's
   json_output['results'][0]


View the Output Display from the JSON
---------------------------------------

The serialized JSON can be used to reproduce the run in other Python scopes. The full display output 
can be viewed by using the :func:`deepchecks.utils.json_utils.from_json` function
(which calls :meth:`deepchecks.core.SuiteResult.from_json` for a Suite's result 
or :meth:`deepchecks.core.CheckResult.from_json` for a Check's result).


.. code-block:: python

   from deepchecks.utils.json_utils import from_json
   
   from_json(serialized_output)