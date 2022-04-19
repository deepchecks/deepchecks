=====================
Deepchecks Telemetry
=====================
Deepchecks uses telemetry to report *anonymous* usage information. This information is essential to help improve
the usage experience of deepchecks for all users.

You can always opt-out by using the CLI command

.. code-block:: bash

    deepchecks telemetry disable

.. note::

    Currently, opting in and out of telemetry is a global setting applicable to all virtual environments within your
    machine.

Why Deepchecks Uses Telemetry
==============================
**Anonymous** telemetry data allow us to prioritize our research efforts and feature development based on usage.
We collect aggregated information on usage and reliability so that we can ensure a high-quality product.
If we did not have this information, all we really get is pip download statistics and chatting with people directly,
which while being valuable, is not enough to seriously improve the tool as a whole.

What About Sensitive Data?
==========================
Your sensitive data never leaves your machine. We:

- **Never** collect any personally identifiable information.
- **Never** collect anything related to your data.
- **Never** collect details about your model.

What Do We Collect?
===================

Deepchecks collects the following events:

* ``package-import``: when deepchecks is imported.

* While exporting checks/suites outputs:
    * ``check-to-html``: when a check is exported to html.
    * ``suite-to-html``: when a suite is exported to html.
    * ``suite-to-json``: when a suite is exported to json.
    * ``suite-to-wandb``: when a suite is exported to wandb.

* While running checks:
    * ``run-check``: when a check is run.
    * ``run-suite``: when a check fails.

Each event contains the following information:

- A unique UUID that is anonymous.
- The version of **Deepchecks** that is running.
- Operating system information, e.g. Ubuntu 18.04.
- Python version and runtime information, e.g. docker, notebook, etc.

We **cannot identify individual users** from the dataset. It is anonymized and untraceable back to the user.