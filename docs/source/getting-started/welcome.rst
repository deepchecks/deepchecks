.. image:: /_static/images/general/deepchecks-logo-with-white-wide-back.png
   :target: https://deepchecks.com/?utm_source=docs.deepchecks.com/stable&utm_medium=referral&utm_campaign=welcome
   :alt: Deepchecks Logo
   :align: center
   :width: 80%

=========================
Welcome to Deepchecks!
=========================

Deepchecks is a holistic tool for testing, validating and monitoring your machine learning models
and data, throughout the model's lifecycle. It enables you to identify problems with your
data quality, distributions, and model's performance with minimal effort.

See more info in the :ref:`Deepchecks Components for Continuous Validation <welcome__deepchecks_components>`
section, along with the direct links to the documentation of each component.


.. _welcome__start_working:

Get Started with Deepchecks Testing
========================================

.. image:: /_static/images/general/checks-and-conditions.png
   :alt: Deepchecks Testing Suite of Checks
   :width: 65%
   :align: center


.. grid:: 1
    :gutter: 3
   
    .. grid-item-card:: 🏃‍♀️ Quickstarts 🏃‍♀️
        :link-type: ref
        :link: welcome__quickstarts
         
        Downloadable end-to-end guides, demonstrating how to start testing your data & model
        in just a few minutes.

    .. grid-item-card:: 💁‍♂️ Get Help & Give Us Feedback 💁
        :link-type: ref
        :link: welcome__get_help

        Links for how to interact with us via our `Slack Community <https://www.deepchecks.com/slack>`__
        or by opening `an issue on Github <https://github.com/deepchecks/deepchecks/issues>`__.


    .. grid-item-card:: 💻  Install 💻 
        :link-type: doc
        :link: /getting-started/installation

        Full installation guide (quick one can be found in quickstarts)

    .. grid-item-card:: 🤓 General: Concepts & Guides 🤓
        :link-type: ref
        :link: general__index
         
        A comprehensive view of deepchecks concepts,
        customizations, and core use cases.

    .. grid-item-card:: 🔢 Tabular 🔢
        :link-type: ref
        :link: tabular__index

        Quickstarts, main concepts, checks gallery and end-to-end guides demonstrating 
        how to start working Deepchecks with tabular data and models.

    .. grid-item-card:: 🔤️ NLP 🔤️
        :link-type: ref
        :link: nlp__index

        Quickstarts, main concepts, checks gallery and end-to-end guides demonstrating
        how to start working Deepchecks with textual data.
        Future releases to come!

    .. grid-item-card:: 🎦‍ Computer Vision (Note: in Beta Release) 🎦‍
        :link-type: ref
        :link: vision__index
         
        Quickstarts, main concepts, checks gallery and end-to-end guides demonstrating 
        how to start working Deepchecks with CV data and models.
        Built-in support for PyTorch, TensorFlow, and custom frameworks.
    
    .. grid-item-card:: 🚀 Interactive Checks Demo 🚀
        :link-type: url
        :link: https://checks-demo.deepchecks.com/?check=No+check+selected
             &utm_source=docs.deepchecks.com&utm_medium=referral&
             utm_campaign=welcome_page&utm_content=checks_demo_card
      
        Play with some of the existing tabular checks
        and see how they work on various datasets with custom corruptions injected.

    .. grid-item-card:: 🤖 API Reference 🤖
        :link-type: doc
        :link: /api/index

        Reference and links to source code for Deepchecks Testing's components.


.. _welcome__quickstarts:

🏃‍♀️ Testing Quickstarts 🏃‍♀️
==============================

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: 🔢 Tabular 🔢 
        :link-type: doc
        :link: /tabular/auto_tutorials/quickstarts/index
        :columns: 4
    
    .. grid-item-card:: 🎦‍ Vision 🎦‍ (in Beta)
        :link-type: doc
        :link: /vision/auto_tutorials/quickstarts/index
        :columns: 4

    .. grid-item-card:: 🔤️ NLP 🔤️ (in Alpha)
        :link-type: doc
        :link: /nlp/auto_tutorials/quickstarts/plot_text_classification
        :columns: 4



.. _welcome__deepchecks_components:

Deepchecks' Components
=======================

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: Testing Docs (Here)
        :link-type: ref
        :link: welcome__start_working
        :img-top: /_static/images/welcome/research_title.png
        :columns: 4

        Tests during research and model development
    
    .. grid-item-card:: CI Docs
        :link-type: doc
        :link: /general/usage/ci_cd
        :img-top: /_static/images/welcome/ci_cd_title.png
        :columns: 4
        
        Tests before deploying the model to production

    .. grid-item-card:: Monitoring Docs
        :link-type: ref
        :link: deepchecks-mon:welcome__start_with_deepchecks_monitoring
        :img-top: /_static/images/welcome/monitoring_title.png
        :columns: 4

        Tests and continuous monitoring during production

Deepchecks accompanies you through various testing needs such as verifying your data's integrity, 
inspecting its distributions, validating data splits, evaluating your model and comparing between different models,
throughout the model's lifecycle.

.. image:: /_static/images/welcome/testing_phases_in_pipeline.png
   :alt: Phases for Continuous Validation of ML Models and Data
   :align: center

|

Deechecks' continuous validation approach is based on testing the ML models and data throughout the different phases 
using the exact same checks, enabling a simple, elaborate and seamless experience for configuring and consuming the results.
Each phase has its relevant interfaces (e.g. visual outputs, output results to json, alert configuration, push notifications, RCA, etc.) for
interacting with the test results.

.. _welcome__get_help:

Get Help & Give Us Feedback
============================

.. admonition:: Join Our Community 👋
   :class: tip

   In addition to perusing the documentation, feel free to:

   - Ask questions on the `Slack Community <https://www.deepchecks.com/slack>`__.
   - Post an issue or start a discussion on `Github Issues <https://github.com/deepchecks/deepchecks/issues>`__.
   - To contribute to the package, check out the 
     `Contribution Guidelines <https://github.com/deepchecks/deepchecks/blob/main/CONTRIBUTING.rst>`__ and join the 
     `contributors-q-and-a channel <https://deepcheckscommunity.slack.com/archives/C030REPARGR>`__ on Slack, 
     or communicate with us via github issues.

   To support us, please give us a star on ⭐️ `Github <https://github.com/deepchecks/deepchecks>`__ ⭐️, 
   it really means a lot for open source projects!
