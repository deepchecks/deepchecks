.. image:: /_static/images/general/deepchecks-logo-with-white-wide-back.png
   :target: https://deepchecks.com/?utm_source=docs.deepchecks.com&utm_medium=referral&utm_campaign=welcome
   :alt: Deepchecks Logo
   :align: center

========================
Welcome to Deepchecks!
========================

Deepchecks is the leading tool for testing, validating and 
:doc:`monitoring <deepchecks-mon:getting-started/welcome>` your machine learning models
and data, and it enables doing so with minimal effort. Deepchecks accompanies you through
various validation and testing needs such as verifying your data's integrity, inspecting its distributions,
validating data splits, evaluating your model and comparing between different models.

.. image:: /_static/images/general/checks-and-conditions.png
   :alt: Deepchecks Suite of Checks
   :width: 75%
   :align: center

|

.. _welcome__start_working:

Start Working with Deepchecks Testing
==========================================

.. grid:: 1
    :gutter: 3
   
    .. grid-item-card:: 🤓 Concepts & Guides 🤓
        :link-type: doc
        :link: /general/index
         
        A comprehensive view of deepchecks concepts,
        customizations, and core use cases.
   
    .. grid-item-card:: 💻  Install 💻 
        :link-type: doc
        :link: /getting-started/installation

        Full installation guide (quick one can be found in quickstarts)

    .. grid-item-card:: 🚀 Interactive Checks Demo 🚀
        :link-type: url
        :link: https://checks-demo.deepchecks.com/?check=No+check+selected
             &utm_source=docs.deepchecks.com&utm_medium=referral&
             utm_campaign=welcome_page&utm_content=checks_demo_card
      
        Play with some of the existing tabular checks
        and see how they work on various datasets with custom corruptions injected.

    .. grid-item-card:: 📋 Tabular 📋‍
        :link-type: doc
        :link: /tabular/index

        Main concepts, check gallery and end-to-end guides demonstrating how to start working Deepchecks
        with tabular data and models.

    .. grid-item-card:: 🎦‍ Computer Vision (Note: in Beta Release) 🎦‍
        :link-type: doc
        :link: /vision/index
         
        Main concepts, check gallery and end-to-end guides demonstrating how to start working Deepchecks
        with CV data and models. Build-in support for PyTorch, TensorFlow, and custom frameworks.

    .. grid-item-card:: 🔤️ NLP (Note: in Alpha Release) 🔤️
        :link-type: doc
        :link: /nlp/index

        Main concepts, check gallery and end-to-end guides demonstrating how to start working Deepchecks
        with NLP data.

    .. grid-item-card:: 💁‍♂️ Get Help & Give Us Feedback 💁
        :link-type: ref
        :link: welcome__get_help

        Links for how to interact with us via our `Slack Community  <https://www.deepchecks.com/slack>`__
        or by opening `an issue on Github <https://github.com/deepchecks/deepchecks/issues>`__.

    .. grid-item-card:: 🤖 API Reference 🤖
        :link-type: doc
        :link: /api/index

        Reference and links to source code for Deepchecks' components

.. _welcome__get_help:

Get Help & Give Us Feedback
============================

.. admonition:: Join Our Community 👋
   :class: tip

   In addition to perusing the documentation, feel free to:

   - Ask questions on our `Slack Community <https://www.deepchecks.com/slack>`__,
   - Post an issue or start a discussion on `Github Issues <https://github.com/deepchecks/deepchecks/issues>`__.

   To support us, please give us a star ⭐️ on `Github <https://github.com/deepchecks/deepchecks>`__, it really means a lot for open source projects!

Deepchecks' Components
=======================

Continuous validation of ML models and data includes testing throughout the model's lifecycle:

.. image:: /_static/images/welcome/testing_phases_in_pipeline.png
   :alt: Phases for Continuous Validation of ML Models and Data
   :align: center

|

Head over to the relevant documentation for more info:

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: Testing Package (Here)
        :link-type: ref
        :link: welcome__start_working
        :img-top: /_static/images/welcome/research_title.png
        :columns: 4

        Tests during research and model development
    
    .. grid-item-card:: Testing Package CI/CD Usage
        :link-type: doc
        :link: /general/usage/ci_cd
        :img-top: /_static/images/welcome/ci_cd_title.png
        :columns: 4
        
        Tests before deploying the model to production

    .. grid-item-card:: Monitoring
        :link-type: ref
        :link: deepchecks-mon:welcome__start_with_deepchecks_monitoring
        :img-top: /_static/images/welcome/monitoring_title.png
        :columns: 4

        Tests and continuous monitoring during production



