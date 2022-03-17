.. raw:: html

   <!--
     ~ ----------------------------------------------------------------------------
     ~ Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
     ~
     ~ This file is part of Deepchecks.
     ~ Deepchecks is distributed under the terms of the GNU Affero General
     ~ Public License (version 3 or later).
     ~ You should have received a copy of the GNU Affero General Public License
     ~ along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
     ~ ----------------------------------------------------------------------------
     ~
   -->

.. raw:: html

   <p align="center">
     &emsp;
     <a href="https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg">Join&nbsp;Slack</a>
     &emsp; | &emsp; 
     <a href="https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=top_links">Documentation</a>
     &emsp; | &emsp; 
     <a href="https://deepchecks.com/blog/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=top_links">Blog</a>
     &emsp; | &emsp;  
     <a href="https://twitter.com/deepchecks">Twitter</a>
     &emsp;
   </p>
   
.. raw:: html

   <p align="center">
      <a href="https://deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=logo">
      <img src="/docs/images/deepchecks-logo-with-white-wide-back.png">
      </a>
   </p>

|build| |Documentation Status| |pkgVersion| |pyVersions|
|Maintainability| |Coverage Status|

.. raw:: html

   <h1 align="center">
      Testing and Validating ML Models & Data
   </h1>

.. raw:: html

   <p align="center">
      <img src="/docs/images/checks-and-conditions.png">
   </p>


🧐 What is Deephchecks?
==========================

Deepchecks is a Python package for comprehensively validating your
machine learning models and data with minimal effort. This includes
checks related to various types of issues, such as model performance,
data integrity, distribution mismatches, and more.


🖼️ Computer Vision & 🔢 Tabular Support
==========================================
**This README refers to the Computer Vision & Images** subpackage of deepchecks which is currently in beta release.
For an overview of the deepchecks package and more details about the Tabular version, `go here <https://github.com/deepchecks/deepchecks>`__.


💻 Installation
=================

Using pip
----------

.. code:: bash

   pip install deepchecks[vision] -U --user

Using conda
------------

.. code:: bash

   conda install -c conda-forge deepchecks[vision]


⏩ Try it Out!
===============

Check out our `tutorials for a quick intro to deepchecks for CV <https://docs.deepchecks.com/en/stable/
tutorials/vision/?
utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out>>`.


📊 Check Examples
==================

To run a specific single check, all you need to do is import it and then
to run it with the required (check-dependent) input parameters. More
details about the existing checks and the parameters they can receive
can be found in our `API Reference`_.

.. _API Reference:
   https://docs.deepchecks.com/en/stable/
   api/index.html?
   utm_source=github.com&utm_medium=referral&
   utm_campaign=readme&utm_content=running_a_check

Performance Checks
-------------------

.. code:: python

   from deepchecks.vision.checks import MeanAveragePrecisionReport
   from deepchecks.vision.datasets.detection import coco

   # Initialize and run desired check
   result = MeanAveragePrecisionReport().run(test_ds, yolo)
   result

Will produce output of the type:

   .. raw:: html

      <h4>Mean Average Precision Report</h4>
      <p>Summarize mean average precision metrics on a dataset
      and model per IoU and area range.</p>
      <a href="https://docs.deepchecks.com/en/0.5.0.dev2/examples/vision/checks/performance/mean_average_precision_report.html
      utm_source=display_output&utm_medium=referral&utm_campaign=readme_check_link" target="_blank">
      Read More...</a>
      <h5>Additional Outputs</h5>
      <table id="T_908a2_">
      <thead>
            <tr>
            <th class="blank level0">&nbsp;</th>
            <th class="col_heading level0 col0">mAP@0.5..0.95 (%)</th>
            <th class="col_heading level0 col1">AP@.50 (%)</th>
            <th class="col_heading level0 col2">AP@.75 (%)</th>
            </tr>
            <tr>
            <th class="index_name level0">Area size</th>
            <th class="blank col0">&nbsp;</th>
            <th class="blank col1">&nbsp;</th>
            <th class="blank col2">&nbsp;</th>
            </tr>
      </thead>
      <tbody>
            <tr>
            <th id="T_908a2_level0_row0" class="row_heading level0 row0">All</th>
            <td id="T_908a2_row0_col0" class="data row0 col0">0.41</td>
            <td id="T_908a2_row0_col1" class="data row0 col1">0.57</td>
            <td id="T_908a2_row0_col2" class="data row0 col2">0.43</td>
            </tr>
            <tr>
            <th id="T_908a2_level0_row1" class="row_heading level0 row1">Small (area &lt; 32^2)</th>
            <td id="T_908a2_row1_col0" class="data row1 col0">0.21</td>
            <td id="T_908a2_row1_col1" class="data row1 col1">0.34</td>
            <td id="T_908a2_row1_col2" class="data row1 col2">0.21</td>
            </tr>
            <tr>
            <th id="T_908a2_level0_row2" class="row_heading level0 row2">Medium (32^2 &lt; area &lt; 96^2)</th>
            <td id="T_908a2_row2_col0" class="data row2 col0">0.38</td>
            <td id="T_908a2_row2_col1" class="data row2 col1">0.60</td>
            <td id="T_908a2_row2_col2" class="data row2 col2">0.35</td>
            </tr>
            <tr>
            <th id="T_908a2_level0_row3" class="row_heading level0 row3">Large (area &lt; 96^2)</th>
            <td id="T_908a2_row3_col0" class="data row3 col0">0.54</td>
            <td id="T_908a2_row3_col1" class="data row3 col1">0.67</td>
            <td id="T_908a2_row3_col2" class="data row3 col2">0.59</td>
            </tr>
      </tbody>
      </table>
      <p align="left">
        <img src="/docs/images/mAP-over-IoU.png">
      </p>

Property Distribution Checks
----------------------------

TBD


What Do You Need in Order To Start Validating?
================================================

- Images (optional: model, predictions and labels)
- Pytorch environment
- Supported use cases
    - All use cases are supported for checks that require only the images (e.g. for checking image properties such as brightness or aspect ratio)
    - Checks that require the predictions, labels, and/or that calculate metrics
      have built-in metrics and format support for Object Detection and Classification.
    - Support for additional use cases (segmentation, regression, pose estimation and more) can be added with custom code.


📖 Documentation
====================

-  `https://docs.deepchecks.com/ <https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation>`__
   - HTML documentation (stable release)
-  `https://docs.deepchecks.com/en/latest <https://docs.deepchecks.com/en/latest/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation>`__
   - HTML documentation (latest release)

👭 Community
================

-  Join our `Slack
   Community <https://join.slack.com/t/deepcheckscommunity/shared_invite/zt-y28sjt1v-PBT50S3uoyWui_Deg5L_jg>`__
   to connect with the maintainers and follow users and interesting
   discussions
-  Post a `Github
   Issue <https://github.com/deepchecks/deepchecks/issues>`__ to suggest
   improvements, open an issue, or share feedback.


.. |build| image:: https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg
.. |Documentation Status| image:: https://readthedocs.org/projects/deepchecks/badge/?version=stable
   :target: https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=badge
.. |pkgVersion| image:: https://img.shields.io/pypi/v/deepchecks
.. |pyVersions| image:: https://img.shields.io/pypi/pyversions/deepchecks
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/970b11794144139975fa/maintainability
   :target: https://codeclimate.com/github/deepchecks/deepchecks/maintainability
.. |Coverage Status| image:: https://coveralls.io/repos/github/deepchecks/deepchecks/badge.svg?branch=main
   :target: https://coveralls.io/github/deepchecks/deepchecks?branch=main