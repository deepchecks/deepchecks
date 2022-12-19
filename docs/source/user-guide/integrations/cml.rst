============================
CML
============================

`CML <https://cml.dev>`__ is a CLI from from the creators of DVC  - Iterative AI
- that helps integrate your machine learning projects in your CI pipeline.

Deepchecks has an option to save the results of a suite
as a summary markdown that includes the full html report
as an attachment - as GitHub markdown and GitLab markdown do not run javascript.

The example here is written for GitLab CI, but the same principles apply in other CI systems.



Export SuiteResult as a Markdown and HTML files
-----------------------------------------------

.. code:: ipython3

    from deepchecks.tabular.datasets.classification.adult import load_data, load_fitted_model
    from deepchecks.tabular.suites import train_test_validation
    model = load_fitted_model()
    train, test = load_data()
    ttvs = train_test_validation()

    # run the suite and get a SuiteResult
    result = ttvs.run(train_dataset=train, test_dataset=test, model=model)

    # save the SuiteResult as a GitLab- or GitHub- compliant markdown
    result.save_as_cml_markdown(file='report_gitlab.md', format='gitlab')
    # a full html report - report_gitlab.html - is produced alongside report_gitlab.md
    # its relative path to the .md file must stay consistent for cml to find it.


Use CML post the report to a Pull/Merge Request
-----------------------------------------------

.. code:: yaml

    test-data-integrity:
      stage: test_data
      script:
        - dvc pull
        # say this command produces ./report_gitlab.md and ./report_gitlab.html
        - dvc repro train_test_validation
        # make cml make a comment in the PR/MR with the produced summary report file
        - cml comment create report_gitlab.md --publish --publish-native
