# Contributing Guidelines

## Pull Request Checklist
- Read the [contributing guidelines](https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.md).
- Check if your changes are consistent with the [guidelines](https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [coding style](https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.md#coding-style)
- Run the [unit tests](https://github.com/deepchecks/deepchecks/blob/master/CONTRIBUTING.md#running-unit-tests).




#### General guidelines for contribution

-   Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
-   Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.
-   Keep API compatibility in mind when you change code. Reviewers of your pull request will comment on any API compatibility issues.

#### Coding Style
Changes to Python code should pass both linting and docstring check,
to make it easier with our contributors, we've included a `makefile` which help you get your code style on point.
in order to validate that your code style, you can run
 ```bash
make validate
``` 
which in turn will run `pylint` and `pydocstring` on the code.

#### Running Unit Tests
Every Pull Request Submitted will be checked on every supported Python version,
in your on-going development, you can fall back to `make test` in order to check your tests on a single python version.
when finishing with your development and prior to creating a pull request, run `make tox` which in turn will run the tests on every supported python version, thus validating that your PR tests will pass.

