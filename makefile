# This makefile helps with deepchecks Development environment
# including syntax checking, virtual environments creation,
# test running and coverage
# This Makefile is based on Makefile by jidn: https://github.com/jidn/python-Makefile/blob/master/Makefile

# Package = Source code Directory
PACKAGE = deepchecks

# Requirements dir & requirements suffix file
REQUIRE_DIR = ./requirements
REQUIRE_FILE = requirements.txt

# python3 binary takes precedence over python binary,
# this variable is used when setting python variable, (Line 18)
# and on 'env' goal ONLY
# If your python path binary name is not python/python3,
# override using ext_python=XXX and it'll propagate into python variable, too
ext_py := $(shell which python3 || which python)

# Override by putting in commandline python=XXX when needed.
python = $(shell echo ${ext_py} | rev | cut -d '/' -f 1 | rev)
TESTDIR = $(shell realpath tests)
ENV = $(shell realpath venv)
repo = pypi

WIN_ENV := venv
WIN_TESTDIR := tests
WIN_BIN := $(WIN_ENV)/bin

# System Envs
BIN := $(ENV)/bin
pythonpath := PYTHONPATH=.
OS := $(shell uname -s)

# Venv Executables
PIP := $(BIN)/pip
PIP_WIN := python -m pip
PYTHON := $(BIN)/$(python)
ANALIZE := $(BIN)/pylint -j 0
COVERAGE := $(BIN)/coverage
COVERALLS := $(BIN)/coveralls
FLAKE8 := $(BIN)/flake8 --whitelist spelling-allowlist.txt --exclude=deepchecks/vision/datasets/assets,deepchecks/ppscore.py
FLAKE8_RST := $(BIN)/flake8-rst
PYTEST := $(BIN)/pytest
TOX := $(BIN)/tox
TWINE := $(BIN)/twine
APIDOC := $(BIN)/sphinx-apidoc
SPHINX_BUILD := $(BIN)/sphinx-build
JUPYTER := $(BIN)/jupyter
LYCHEE := $(BIN)/lychee

# Project Settings
PKGDIR := $(or $(PACKAGE), ./)
SOURCES := $(or $(PACKAGE), $(wildcard *.py))


# Test and Analyize
TEST_CODE := tests/

PYLINT_LOG = .pylint.log

# Coverage vars
COVERAGE_LOG = .cover.log
COVERAGE_FILE = default.coveragerc
COVERAGE_RC := $(wildcard $(COVERAGE_FILE))
COVER_ARG := --cov-report term-missing --cov=$(PKGDIR) \
	$(if $(COVERAGE_RC), --cov-config $(COVERAGE_RC))


# Documentation
#
DOCS         := $(shell realpath ./docs)
DOCS_SRC     := $(DOCS)/source
DOCS_BUILD   := $(DOCS)/build
DOCS_REQUIRE := $(DOCS)/$(REQUIRE_FILE)

# variables that will be passed to the documentation make file
SPHINXOPTS   ?=


EGG_INFO := $(subst -,_,$(PROJECT)).egg-info
EGG_LINK = venv/lib/python3.7/site-packages/deepchecks.egg-link


### Main Targets ######################################################

.PHONY: help env all requirements doc-requirements dev-requirements

# TODO: add description for all targets (at least for the most usefull)

help:
	@echo "env"
	@echo ""
	@echo "    Create virtual environment and install requirements"
	@echo "    python=PYTHON_EXE interpreter to use, default=python,"
	@echo "    when creating new env and python binary is 2.X, use 'make env python=python3'"
	@echo ""
	@echo "validate"
	@echo ""
	@echo "    Run style checks 'pylint' , 'docstring'"
	@echo "    pylint docstring - sub commands of validate"
	@echo ""
	@echo "test"
	@echo ""
	@echo "    TEST_RUNNER on '$(TESTDIR)'"
	@echo "    args=\"<pytest Arguements>\" optional arguments"
	@echo ""
	@echo "coverage"
	@echo ""
	@echo "    Get coverage information, optional 'args' like test"
	@echo ""
	@echo "jupyter"
	@echo ""
	@echo "    Deploy jupyer-notebook using './examples' directory"
	@echo "    args=\"<jupyter Arguments\" -passable"
	@echo ""
	@echo "tox"
	@echo ""
	@echo "    Test against multiple versions of python as defined in tox.ini"
	@echo ""
	@echo "clean | clean-all"
	@echo ""
	@echo "    Clean up | clean up & removing virtualenv"
	@echo ""
	@echo "docs"
	@echo ""
	@echo "    Build documentation site content"
	@echo ""
	@echo "show-docs"
	@echo ""
	@echo "    Show documentation in the browser"
	@echo ""
	@echo "trailing-spaces"
	@echo ""
	@echo "    Remove trailing whitespaces from all python modules"
	@echo ""
	



all: validate test


env: $(ENV)


$(ENV):
	@echo "#### Creating Python Virtual Environment [ $(ENV) ] ####"
	@echo "external python_exe is $(ext_py)"
	@test -d $(ENV) || $(ext_py) -m venv $(ENV)


vision-torch-tf-setup: env
	@echo "####  installing torch and tensorflow packages #### "

	@if [ -x "$$(command -v nvidia-smi)" ]; \
	then \
		$(PIP) install -q \
			"torch==2.4.1" "torchvision==0.19.1" \
			--index-url https://download.pytorch.org/whl/cu118; \
		$(PIP) install -q "tensorflow-gpu==2.11.0"; \
	elif [ $(OS) = "Linux" ]; \
	then \
		$(PIP) install -q \
			"torch==2.4.1" "torchvision==0.19.1" \
			--index-url https://download.pytorch.org/whl/cpu; \
		$(PIP) install -q "tensorflow==2.11.0"; \
	else \
		$(PIP) install -q \
			"torch==2.4.1" "torchvision==0.19.1" \
			--index-url https://download.pytorch.org/whl/cpu; \
		$(PIP) install -q "tensorflow==2.11.0"; \
	fi;

	@$(PIP) install -q "tensorflow-hub==0.12.0";

requirements: vision-torch-tf-setup
	@echo "####  installing dependencies, it could take some time, please wait! #### "
	@$(PIP) install -U pip
	@$(PIP) install wheel setuptools setuptools_scm
	@$(PIP) install -q \
		-r $(REQUIRE_DIR)/$(REQUIRE_FILE) \
		-r $(REQUIRE_DIR)/vision-$(REQUIRE_FILE) \
		-r $(REQUIRE_DIR)/nlp-$(REQUIRE_FILE) \
		-r $(REQUIRE_DIR)/nlp-prop-$(REQUIRE_FILE)
	@$(PIP) install --no-deps -e .

vision-requirements: vision-torch-tf-setup
	@echo "####  installing dependencies, it could take some time, please wait! #### "
	@$(PIP) install -U pip
	@$(PIP) install wheel setuptools setuptools_scm
	@$(PIP) install -q \
		-r $(REQUIRE_DIR)/$(REQUIRE_FILE) \
		-r $(REQUIRE_DIR)/vision-$(REQUIRE_FILE)
	@$(PIP) install --no-deps -e .

doc-requirements: $(ENV)
	@echo "####  installing documentation dependencies, it could take some time, please wait! #### "
	@$(PIP) install -q -r ./docs/requirements.txt


dev-requirements: $(ENV)
	@echo "####  installing development dependencies, it could take some time, please wait! ####"
	@$(PIP) install -q -r $(REQUIRE_DIR)/dev-$(REQUIRE_FILE)

dev-nlp-requirements: $(ENV)
	@echo "####  installing nlp development dependencies, it could take some time, please wait! ####"
	@$(PIP) install -q -r $(REQUIRE_DIR)/dev-nlp-$(REQUIRE_FILE)
	@$(PIP) install -q \
			"torch==2.4.1" "torchvision==0.19.1" \
			--index-url https://download.pytorch.org/whl/cpu;

### Static Analysis ######################################################

.PHONY: validate pylint docstring


validate: pylint docstring


pylint: dev-requirements
	$(ANALIZE) $(SOURCES)
	$(ANALIZE) $(TEST_CODE) --ignore-paths='.*\/checks\/.+$\'
	$(FLAKE8) $(SOURCES)
	$(FLAKE8_RST) $(SOURCES)


docstring: dev-requirements
	$(PYTHON) -m pydocstyle --convention=pep257 --add-ignore=D107 $(SOURCES)


### Testing ######################################################

.PHONY: test coverage


test: requirements dev-requirements dev-nlp-requirements
	@if [ ! -z $(args) ]; then \
		$(PYTEST) $(args); \
	else \
		$(PYTEST) $(TESTDIR); \
	fi;


vision-gpu-tests: vision-requirements dev-requirements
	$(PYTEST) $(TESTDIR)/vision/gpu_tests

test-win:
	@test -d $(WIN_ENV) || python -m venv $(WIN_ENV)
	@$(WIN_ENV)\Scripts\activate.bat
	@$(PIP_WIN) install -U pip wheel setuptools setuptools_scm
	@$(PIP_WIN) install -q \
			"torch==2.4.1" "torchvision==0.19.1" \
			--index-url https://download.pytorch.org/whl/cpu;
	@$(PIP_WIN) install -q "tensorflow-hub==0.12.0";
	@$(PIP_WIN) install -q "tensorflow==2.11.0";
	@$(PIP_WIN) install -U pip
	@$(PIP_WIN) install -q \
		-r $(REQUIRE_DIR)/$(REQUIRE_FILE)  \
		-r $(REQUIRE_DIR)/vision-$(REQUIRE_FILE)  \
		-r $(REQUIRE_DIR)/nlp-$(REQUIRE_FILE)  \
		-r $(REQUIRE_DIR)/nlp-prop-$(REQUIRE_FILE)  \
		-r $(REQUIRE_DIR)/dev-$(REQUIRE_FILE) \
		-r $(REQUIRE_DIR)/dev-nlp-$(REQUIRE_FILE)
	@$(PIP_WIN) install -e .
	python -m pytest -vvv $(WIN_TESTDIR)


test-tabular-only: env
	@$(PIP) install -U pip wheel setuptools
	@$(PIP) install \
		-r $(REQUIRE_DIR)/$(REQUIRE_FILE) \
		-r $(REQUIRE_DIR)/dev-$(REQUIRE_FILE)
	@$(PIP) install --no-deps -e .
	$(PYTEST) -vvv $(TESTDIR)/tabular


coverage: requirements dev-requirements dev-nlp-requirements
	$(COVERAGE) run --source deepchecks/,tests/ --omit */assets/* -m pytest

coveralls: coverage
	$(COVERALLS) --service=github


# This is Here For Legacy || future use case,
# our PKGDIR is in its own directory so we dont really need to remove the ENV dir.
$(COVERAGE_FILE):
ifeq ($(PKGDIR),./)
ifeq (,$(COVERAGE_RC))
	# If PKGDIR is root directory, ie code is not in its own directory
	# then you should use a .coveragerc file to remove the ENV directory
	$(info Rerun make to discover autocreated $(COVERAGE_FILE))
	@echo -e "[run]\nomit=$(ENV)/*" > $(COVERAGE_FILE)
	@cat $(COVERAGE_FILE)
	@exit 68
endif
endif


tox: requirements dev-requirements
	$(TOX)


freeze: requirements dev-requirements dev-nlp-requirements
	@$(PIP) freeze


### Cleanup ######################################################

.PHONY: clean clean-env clean-all clean-build clean-test clean-dist clean-docs trailing-spaces


clean: clean-dist clean-test clean-build clean-docs


clean-all: clean clean-env


clean-env: clean
	-@rm -rf $(ENV)
	-@rm -rf $(COVERAGE_LOG)
	-@rm -rf $(PYLINT_LOG)
	-@rm -rf ./lychee.output
	-@rm -rf .tox


clean-build:
	@find $(PKGDIR) -name '*.pyc' -delete
	@find $(PKGDIR) -name '__pycache__' -delete
	@find $(TESTDIR) -name '*.pyc' -delete 2>/dev/null || true
	@find $(TESTDIR) -name '__pycache__' -delete 2>/dev/null || true
	-@rm -rf $(EGG_INFO)
	-@rm -rf __pycache__


clean-test:
	-@rm -rf .pytest_cache
	-@rm -rf .coverage


clean-dist:
	-@rm -rf dist build


clean-docs: $(DOCS)
	@rm -rf $(DOCS_BUILD)
	@rm -rf $(DOCS)/docs.error.log


trailing-spaces:
	@find ./deepchecks/ -name "*.py" -type f -print0 | xargs -0 sed -i "s/[[:space:]]*$$//"
	@find ./tests/ -name "*.py" -type f -print0 | xargs -0 sed -i "s/[[:space:]]*$$//"


### Release ######################################################

.PHONY: authors register dist upload test-upload release test-release .git-no-changes


authors:
	echo "Authors\n=======\n\nA huge thanks to all of our contributors:\n\n" > AUTHORS.md
	git log --raw | grep "^Author: " | cut -d ' ' -f2- | cut -d '<' -f1 | sed 's/^/- /' | sort | uniq >> AUTHORS.md


dist: $(ENV)
	$(PIP) install wheel twine
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel


# upload expects to get all twine args as environment,
# refer to https://twine.readthedocs.io/en/latest/ for more information
#
upload: $(TWINE)
	$(TWINE) upload dist/*


# TestPyPI – a separate instance of the Python Package Index that allows us
# to try distribution tools and processes without affecting the real index.
#
test-upload: $(TWINE)
	$(TWINE) upload --repository-url https://test.pypi.org/legacy/ dist/*


release: dist upload
test-release: dist test-upload


.git-no-changes:
	@if git diff --name-only --exit-code;       \
	then                                        \
		echo Git working copy is clean...;        \
	else                                        \
		echo ERROR: Git working copy is dirty!;   \
		echo Commit your changes and try again.;  \
		exit -1;                                  \
	fi;


### Documentation

.PHONY: docs validate-examples website dev-docs gen-static-notebooks license-check links-check


docs: requirements doc-requirements dev-requirements dev-nlp-requirements develop $(DOCS_SRC)
	@export WANDB_MODE=offline
	cd $(DOCS) && make html SPHINXBUILD=$(SPHINX_BUILD) SPHINXOPTS=$(SPHINXOPTS)
	@echo ""
	@echo "++++++++++++++++++++++++"
	@echo "++++ Build Finished ++++"
	@echo "++++++++++++++++++++++++"
	@echo ""
	@echo "all errors/warnings were written to the file:"
	@echo "- $(DOCS)/docs.error.log"
	@echo ""
	@echo "statistic:"
	@echo "- ERRORs: $$(grep "ERROR" $(DOCS)/docs.error.log | wc -l)"
	@echo "- WARNINGs: $$(grep "WARNING" $(DOCS)/docs.error.log | wc -l)"

validate-examples: doc-requirements
	@$(PYTHON) $(TESTDIR)/examples_validation.py

show-docs: $(DOCS_BUILD)/html
	@cd $(DOCS_BUILD)/html && $(PYTHON) -m http.server


license-check:
	@wget https://dlcdn.apache.org/skywalking/eyes/0.7.0/skywalking-license-eye-0.7.0-bin.tgz && tar -xzvf skywalking-license-eye-0.7.0-bin.tgz
	@mv skywalking-license-eye-0.7.0-bin/bin/linux/license-eye ./
	@rm -rf skywalking-license-eye-0.7.0-bin && rm -f skywalking-license-eye-0.7.0-bin.tgz
	./license-eye -c .licenserc_fix.yaml header check
	@rm license-eye


links-check: $(DOCS_BUILD) $(LYCHEE)
	@$(LYCHEE) \
		"./deepchecks/**/*.rst" "./*.rst" "$(DOCS_BUILD)/html/**/*.html" \
		--base $(DOCS_BUILD)/html \
		--accept=200,403,429 \
		--format markdown \
		--output ./lychee.output \
		--exclude-loopback \
		--exclude-mail \
		--exclude-file $(DOCS)/.lycheeignore \
		--exclude ".*git.*"; \
	if [ $? -eq 0 ]; \
	then \
		echo "+++ Nothing Detected +++"; \
		exit 0; \
	else \
		echo ""; \
		echo "++++++++++++++++++++++++++++"; \
		echo "++++ Links Check Failed ++++"; \
		echo "++++++++++++++++++++++++++++"; \
		echo ""; \
		echo "full output was written to the next file:"; \
		echo "- $(shell realpath ./lychee.output)"; \
		echo ""; \
		head -n 12 lychee.output; \
		exit 1; \
	fi;


$(LYCHEE):
	@curl -L --output lychee.tar.gz https://github.com/lycheeverse/lychee/releases/download/v0.8.2/lychee-v0.8.2-x86_64-unknown-linux-gnu.tar.gz
	@tar -xvzf lychee.tar.gz
	@rm -rf ./lychee.tar.gz
	@chmod +x ./lychee
	@mkdir -p $(BIN)/
	@mv ./lychee $(BIN)/


### System Installation ######################################################

.PHONY: develop install download jupyter

develop:
	$(PIP) install -e .

install:
	$(PIP) install .

download:
	$(PIP) install $(PROJECT)

jupyter: $(JUPYTER)
	$(BIN)/jupyter-notebook $(args) --notebook-dir=./docs/source/examples

$(JUPYTER):
	$(PIP) install jupyter