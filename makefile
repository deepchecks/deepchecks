# This makefile helps with deepchecks Development environment
# including syntax checking, virtual environments creation,
# test running and coverage
# This Makefile is based on Makefile by jidn: https://github.com/jidn/python-Makefile/blob/master/Makefile

# Package = Source code Directory
PACKAGE = deepchecks

# Requirements file
REQUIRE = requirements.txt

# python3 binary takes predecence over python binary,
# this variable is used when setting python variable, (Line 18)
# and on 'env' goal ONLY
# If your python path binary name is not python/python3,
# override using ext_python=XXX and it'll propogate into python variable, too
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

# Venv Executables
PIP := $(BIN)/pip
PIP_WIN := python -m pip
PYTHON := $(BIN)/$(python)
ANALIZE := $(BIN)/pylint
COVERAGE := $(BIN)/coverage
COVERALLS := $(BIN)/coveralls
FLAKE8 := $(BIN)/flake8 --whitelist spelling-allowlist.txt
FLAKE8_RST := $(BIN)/flake8-rst
TEST_RUNNER := $(BIN)/pytest
TOX := $(BIN)/tox
TWINE := $(BIN)/twine
APIDOC := $(BIN)/sphinx-apidoc
SPHINX_BUILD := $(BIN)/sphinx-build
JUPYTER := $(BIN)/jupyter
LYCHEE := $(BIN)/lychee

# Project Settings
PKGDIR := $(or $(PACKAGE), ./)
SOURCES := $(or $(PACKAGE), $(wildcard *.py))

# Installation packages
INSTALLATION_PKGS = wheel setuptools

REQUIREMENTS := $(shell find . -name $(REQUIRE))
REQUIREMENTS_LOG := .requirements.log

# Test and Analyize
ANALIZE_PKGS = pylint pydocstyle flake8 flake8-spellcheck flake8-eradicate flake8-rst
TEST_CODE := tests/
TEST_RUNNER_PKGS = pytest pytest-cov pyhamcrest nbval coveralls torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
NOTEBOOK_CHECKS = ./docs/source/examples/checks
NOTEBOOK_EXAMPLES = ./docs/source/examples/guides/*.ipynb
NOTEBOOK_USECASES = ./docs/source/examples/use-cases/*.ipynb
NOTEBOOK_SANITIZER_FILE = ./docs/source/examples/.nbval-sanitizer

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
DOCS_REQUIRE := $(DOCS)/$(REQUIRE)

# variables that will be passed to the documentation make file
SPHINXOPTS   ?=


# Sphinx
# SPHINX_PKGS = sphinx pydata-sphinx-theme sphinx-markdown-builder sphinx-autoapi sphinx-copybutton nbsphinx

EGG_INFO := $(subst -,_,$(PROJECT)).egg-info
EGG_LINK = venv/lib/python3.7/site-packages/deepchecks.egg-link

### Main Targets ######################################################
.PHONY: help env all

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
	@echo "    Run style checks 'pylint' , 'docstring' and 'notebook'"
	@echo "    pylint docstring notebook - sub commands of validate"
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


all: validate test notebook

env: $(REQUIREMENTS_LOG)

$(PIP):
	$(info #### Remember to source new environment  [ $(ENV) ] ####)
	@echo "external python_exe is $(ext_py)"
	test -d $(ENV) || $(ext_py) -m venv $(ENV)

$(REQUIREMENTS_LOG): $(PIP) $(REQUIREMENTS)
	$(ext_py) -m pip install --upgrade pip
	$(PIP) install $(INSTALLATION_PKGS)
	for f in $(REQUIREMENTS); do \
	  $(PIP) install -r $$f | tee -a $(REQUIREMENTS_LOG); \
	done


### Static Analysis ######################################################

.PHONY: validate pylint docstring

validate: $(REQUIREMENTS_LOG) pylint docstring

pylint: $(ANALIZE)
	$(ANALIZE) $(SOURCES) $(TEST_CODE)
	$(FLAKE8) $(SOURCES)
	$(FLAKE8_RST) $(SOURCES)

docstring: $(ANALIZE) # We Use PEP257 Style Python Docstring
	$(PYTHON) -m pydocstyle --convention=pep257 --add-ignore=D107 $(SOURCES)

$(ANALIZE): $(PIP)
	$(PIP) install --upgrade $(ANALIZE_PKGS) | tee -a $(REQUIREMENTS_LOG)


### Testing ######################################################

.PHONY: test coverage notebook

test: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
	$(pythonpath) $(TEST_RUNNER) $(args) $(TESTDIR)

test-win:
	test -d $(WIN_ENV) || python -m venv $(WIN_ENV)
	$(WIN_ENV)\Scripts\activate.bat
	$(PIP_WIN) $(INSTALLATION_PKGS)
	for f in requirements/requirements.txt; do \
	$(PIP_WIN) install -r $$f | tee -a $(REQUIREMENTS_LOG); \
	done
	$(PIP_WIN) install $(TEST_RUNNER_PKGS)
	python -m pytest $(WIN_TESTDIR)

notebook: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
# if deepchecks is not installed, we need to install it for testing porpuses,
# as the only time you'll need to run make is in dev mode, we're installing
# deepchecks in development mode
	$(PIP) install --no-deps -e .
# Making sure the examples are running, without validating their outputs.
	@NOTEBOOKS=$$(find ./docs/source/examples -name "*.ipynb") \
	&& N_OF_NOTEBOOKS=$$(find ./docs/source/examples -name "*.ipynb" | wc -l) \
	&& echo "+++ Number of notebooks to execute: $$N_OF_NOTEBOOKS +++" \
	&& $(JUPYTER) nbconvert --execute $$NOTEBOOKS --to notebook --stdout > /dev/null
# For now, because of plotly - disabling the nbval and just validate that the notebooks are running
#	$(pythonpath) $(TEST_RUNNER) --nbval $(NOTEBOOK_CHECKS) --sanitize-with $(NOTEBOOK_SANITIZER_FILE)

$(TEST_RUNNER):
	$(PIP) install $(TEST_RUNNER_PKGS) | tee -a $(REQUIREMENTS_LOG)

regenerate-examples: $(REQUIREMENTS_LOG)
	$(PIP) install --no-deps -e .
	$(JUPYTER) nbextension enable --py widgetsnbextension
	for path in $(NOTEBOOK_EXAMPLES) ; do \
	  $(JUPYTER) nbconvert --to notebook --inplace --execute $$path ; \
	done
	for path in $(NOTEBOOK_USECASES) ; do \
	  $(JUPYTER) nbconvert --to notebook --inplace --execute $$path ; \
	done
	for path in $(NOTEBOOK_CHECKS)/**/*.ipynb ; do \
	  $(JUPYTER) nbconvert --to notebook --inplace --execute $$path ; \
	done


coverage: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
	$(COVERAGE) run -m pytest

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


# tox checks for all python versions matrix
tox: $(TOX)
	$(TOX)

$(TOX): $(PIP)
	$(PIP) install tox | tee -a $(REQUIREMENTS_LOG)


### Cleanup ######################################################

.PHONY: clean clean-env clean-all clean-build clean-test clean-dist clean-docs

clean: clean-dist clean-test clean-build clean-docs

clean-env: clean
	-@rm -rf $(ENV)
	-@rm -rf $(REQUIREMENTS_LOG)
	-@rm -rf $(COVERAGE_LOG)
	-@rm -rf $(PYLINT_LOG)
	-@rm -rf .tox

clean-all: clean clean-env

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

clean-docs: $(DOCS) env  $(SPHINX_BUILD)
	@cd $(DOCS) && make clean SPHINXBUILD=$(SPHINX_BUILD) SPHINXOPTS=$(SPHINXOPTS)

### Release ######################################################
.PHONY: authors register dist upload test-upload release test-release .git-no-changes


authors:
	echo "Authors\n=======\n\nA huge thanks to all of our contributors:\n\n" > AUTHORS.md
	git log --raw | grep "^Author: " | cut -d ' ' -f2- | cut -d '<' -f1 | sed 's/^/- /' | sort | uniq >> AUTHORS.md


dist: test
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


$(TWINE): $(PIP)
	$(PIP) install twine


### Documentation
.PHONY: docs website dev-docs gen-static-notebooks license-check links-check lychee-bin

docs: env $(DOCS_SRC)
	cd $(DOCS) && make html SPHINXBUILD=$(SPHINX_BUILD) SPHINXOPTS=$(SPHINXOPTS) 2> docs.error.log
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


show-docs: $(DOCS_BUILD)/html
	@cd $(DOCS_BUILD)/html && $(PYTHON) -m http.server


license-check:
	@wget https://dlcdn.apache.org/skywalking/eyes/0.2.0/skywalking-license-eye-0.2.0-bin.tgz && tar -xzvf skywalking-license-eye-0.2.0-bin.tgz
	@mv skywalking-license-eye-0.2.0-bin/bin/linux/license-eye ./
	@rm -rf skywalking-license-eye-0.2.0-bin && rm -f skywalking-license-eye-0.2.0-bin.tgz
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
	fi;


$(LYCHEE):
	curl -L --output lychee.tar.gz https://github.com/lycheeverse/lychee/releases/download/v0.8.2/lychee-v0.8.2-x86_64-unknown-linux-gnu.tar.gz \
	&& tar -xvzf lychee.tar.gz \
	&& rm -rf ./lychee.tar.gz \
	&& chmod +x ./lychee \
	&& mkdir -p $(BIN)/ \
	&& mv ./lychee $(BIN)/ \


### System Installation ######################################################
.PHONY: develop install download jupyter

develop:
	$(PYTHON) setup.py develop

install:
	$(PYTHON) setup.py install

download:
	$(PIP) install $(PROJECT)

jupyter: $(JUPYTER)
	$(BIN)/jupyter-notebook $(args) --notebook-dir=./docs/source/examples

$(JUPYTER):
	$(PIP) install jupyter
