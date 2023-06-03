<!--
   ~ ----------------------------------------------------------------------------
   ~ Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
   ~
   ~ This file is part of Deepchecks.
   ~ Deepchecks is distributed under the terms of the GNU Affero General
   ~ Public License (version 3 or later).
   ~ You should have received a copy of the GNU Affero General Public License
   ~ along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
   ~ ----------------------------------------------------------------------------
   ~
-->

[![GitHub
stars](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/deepchecks/deepchecks/stargazers/)
![build](https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg)
![pkgVersion](https://img.shields.io/pypi/v/deepchecks)
[![Maintainability](https://api.codeclimate.com/v1/badges/970b11794144139975fa/maintainability)](https://codeclimate.com/github/deepchecks/deepchecks/maintainability)
[![Coverage
Status](https://coveralls.io/repos/github/deepchecks/deepchecks/badge.svg?branch=main)](https://coveralls.io/github/deepchecks/deepchecks?branch=main) 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section --> [![All Contributors](https://img.shields.io/badge/all_contributors-41-orange.svg?style=flat-round)](#https://github.com/deepchecks/deepchecks/blob/main/CONTRIBUTING.rst) <!-- ALL-CONTRIBUTORS-BADGE:END --> 

<!---
![pyVersions](https://img.shields.io/pypi/pyversions/deepchecks)
--->


<h1 align="center">
   Deepchecks: Continuous Validation for AI & ML: Testing, CI & Monitoring
</h1>

Deepchecks is a holistic open-source solution for all of your AI & ML validation needs,
enabling to thoroughly test your data and models from research to production.


<a target="_blank" href="https://deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=logo">
   <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/images/readme/cont_validation_dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/images/readme/cont_validation_light.png">
      <img alt="Deepchecks continuous validation parts." src="docs/source/_static/images//readme/cont_validation_light.png">
   </picture>
</a>

<p align="center">
   &emsp;
   <a href="https://www.deepchecks.com/slack">👋 Join Slack</a>
   &emsp; | &emsp; 
   <a href="https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=top_links">📖 Documentation</a>
   &emsp; | &emsp; 
   <a href="https://deepchecks.com/blog/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=top_links">🌐 Blog</a>
   &emsp; | &emsp;  
   <a href="https://twitter.com/deepchecks">🐦 Twitter</a>
   &emsp;
</p>
   

<!---
## 🧐 What is Deepchecks?
--->
## 🧮 How does it work?

At its core, Deepchecks includes a wide variety of built-in Checks,
for testing all types of data and model related issues.
These checks are implemented fo various models and data types (Tabular, NLP, Vision), 
and can easily be customized and expanded. 

The check results can be used for automatically making informed decisions
regarding your model's production-readiness, and for monitoring it over time when in production.
They can be examined with visual reports (by saving them to an HTML file, or seeing them in Jupyter),
processed with code (using their json output), and inspected and collaborated upon with a Deepchecks' dynamic UI 
(for examining test results and for production monitoring).

<!---
At its core, Deepchecks has a wide variety of built-in Checks and Suites (lists of checks) 
for all data types (Tabular, NLP, Vision), 
These includes checks for validating your model's performance (e.g. identify weak segments), the data's 
distribution (e.g. detect drifts or leakages), data integrity (e.g. find conflicting labels) and more.
These checks results can be run manually (e.g. during research) or trigerred automatically (e.g. during CI
and production monitoring) and enable automatically making informed decisions regarding your model pipelines' 
production-readiness, and behavior over time.
--->

## 🧩 Components

Deepchecks includes:
- **Deepchecks Testing**
  ([docs](https://docs.deepchecks.com/stable/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=components)): 
  - Built-in Checks & Suites for Tabular, NLP & CV (open source)
- **CI & Testing Management**
  ([docs](https://docs.deepchecks.com/stable/general/usage/ci_cd.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=components)):
  - Collaborating over test results and efficient iterations until 
  model is production-ready and can be deployed (open source & managed offering)
- **Deepchecks Monitoring**
  ([docs](https://docs.deepchecks.com/monitoring/stable/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=components)): 
  - Tracking your deployed models behavior when in production (open source & managed offering).
    The client and deployment for monitoring need to be installed in addition to the testing, check out the 

This repo is our main repo as all components use the deepchecks checks in their core.
If you want to see deepchecks monitoring's code, you can check out the 
[deepchecks/monitoring](https://github.com/deepchecks/monitoring) repo.

## ✅ Deepchecks Checks

- Many preimplemented checks for testing issues such as model performance (e.g. identify weak segments), 
  data distribution (e.g. detect drifts or leakages) and data integrity (e.g. find conflicting labels).
- Customizable: each check has many configurable parameters, and custom checks can easily be implemented.
- Can be run manually (during research) or triggered automatically (in CI processes or production monitoring)
- Check results can be analyzed by:
   - Saving to HTML or viewing in Jupyter - for visual output for human analysis
   - JSON output - for processing with code
   - Deepchecks' UI - for dynamic inspection and collaboration (of test results and production monitoring)
- Optional conditions can be added and customized, to automatically validate whether it passed or not
- A list of checks (with optional conditions) can be run together in a "Suite"

<!---
These checks can be run manually (e.g. during research) or automatically triggered (with CI processes or in scheduled runs for production monitoring).
The check's results can be examined with visual reports (by saving them to an HTML file, or seeing them in Jupyter),
processed with code (using their json output), and inspected and colloaborated upon with a dynamic UI 
(for examining test results and for production monitoring).
Optional conditions can be added to each check, to automatically validate whether it passed or not.
--->

## ⏩  Getting Started

### 💻 Installation

#### Deepchecks Testing Installation

```bash
pip install deepchecks -U --user
```

For installing also the nlp submodle, replace ``deepchecks`` with ``"deepchecks[vision]"``
and for installing it with the computer vision submodule, replace it with ``"deepchecks[nlp]"``. 
For installing with conda, similarly use: ``conda install -c conda-forge deepchecks``.

Check out the full installation instructions for deepchecks testing [here](https://docs.deepchecks.com/stable/getting-started/installation.html).

#### Deepchecks Monitoring Installation

If you're using deepchecks also for production monitoring,
you can deploy a hobby instance in one line on Linux/MacOS (Windows is WIP!) with Docker:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/deepchecks/monitoring/main/deploy/deploy-oss.sh)"
```
This will automatically download the necessary dependencies and start the application locally.
Installing our open source service is an excellent way for local usage, note however, 
that it won't scale to support real-time production usage.

Check out the full installation instructions for deepchecks monitoring [here](https://docs.deepchecks.com/monitoring/stable/installation/index.html). 


### 🏃‍♀️ Quickstarts

#### Deepchecks Testing Quickstart

Jump straight over to the respective quickstarts for 
[Tabular](https://docs.deepchecks.com/stable/tabular/auto_tutorials/quickstarts/index.html), 
[NLP](https://docs.deepchecks.com/stable/nlp/auto_tutorials/quickstarts/index.html), or 
[Vision](https://docs.deepchecks.com/stable/vision/auto_tutorials/quickstarts/index.html) 
models and data,
to have it up and running on your data.

You'll then be able to view all the checks that you chose to run on your data,
and inspect their status and results, with the visual output that looks like this:

<p align="center">
   <img src="docs/source/_static/images/general/model_evaluation_suite.gif" width="800">
</p>


#### Deepchecks Monitoring Quickstart

Jump straight over to the 
[monitoring quickstart](https://docs.deepchecks.com/monitoring/stable/user-guide/tabular/auto_quickstarts/plot_quickstart.html)
to have it up and running on your data.

You'll then be able to see the checks results over time, set alerts, and interact
with the dynamic deepchecks UI that looks like this:

<p align="center">
   <img src="docs/source/_static/images/general/monitoring-app-ui.gif" width="800">
</p>

<!---
### 🔡 Supported Data Types

Deepchecks supports: tabular, textual and image data, with the Tabular, NLP and CV modules.
For more info about the supported use cases, refer to the respective docs within [deepchecks testing[()].

#### 💬 Terminology

- **Check**: the core building block of deepchecks.
  Each check enables you to inspect a specific aspect of your data and models (e.g. drift, performance, integrity issues)
  Checks parameters can be customized and also custom checks can be built and run alongside the framework.
- **Condition**: a function that can be added to a Check, which returns a pass ✓, fail ✖ or warning ! result, intended for 
  validating the Check's return value.
- **Suite**: An ordered collection of Checks, that can have conditions added to them.
  The Suite enables displaying a concluding report for all of the Checks
  that ran.

--->

## 📜 Open Source vs Paid 

Deepchecks' projects (``deepchecks/deepchecks`` & ``deepchecks/monitoring``) are open source and are released under [AGPL 3.0](./LICENSE).

The only exception are the deepchecks monitoring components under the 
([backend/deepchecks_monitoring/ee](https://github.com/deepchecks/monitoring/tree/main/backend/deepchecks_monitoring/ee)) 
directory, that are subject to a commercial license (see the license [here](https://deepchecks.com/terms-and-conditions)).
That directory isn't used by default, and is packaged as part of the deepchecks monitoring repository simply to 
support upgrading to the commercial edition without downtime.

Enabling premium features (contained in the `backend/deepchecks_monitoring/ee` directory) with a self-hosted instance requires a Deepchecks license. 
To learn more, [book a demo](https://deepchecks.com/book-demo/) or see our [pricing page](https://deepchecks.com/pricing).

Looking for a 💯% open-source solution for deepcheck monitoring?
Check out the [Monitoring OSS](https://github.com/deepchecks/monitoring-oss) repository, which is purged of all proprietary code and features.

## 👭 Community, Contributing, Docs & Support

Deepchecks is an open source solution. 
We are committed to a transparent development process and highly appreciate any contributions. 
Whether you are helping us fix bugs, propose new features, improve our documentation or spread the word,
we would love to have you as part of our community.

- Give us a ⭐️ github star ⭐️ on the top of this page to support what we're doing,
  it means a lot for open source projects!
- Read our 
  [docs](https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=docs)
  for more info about how to use and customize deepchecks, and for step-by-step tutorials.
- Post a [Github
  Issue](https://github.com/deepchecks/deepchecks/issues) to submit a bug report, feature request, or suggest an improvement.
- To contribute to the package, check out our [first good issues](https://github.com/deepchecks/deepchecks/contribute)
  and [contribution guidelines](CONTRIBUTING.rst), and open a PR.

Join our Slack to give us feedback, connect with the maintainers and fellow users, ask questions, 
get help for package usage or contributions, or engage in discussions about ML testing!

<a href="https://deepchecks.com/slack"><img src="docs/source/_static/images/general/join-our-slack-community.png" width="200"></a>


## ✨ Contributors

Thanks goes to these wonderful people ([emoji
key](https://allcontributors.org/docs/en/emoji-key)):


<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ItayGabbay"><img src="https://avatars.githubusercontent.com/u/20860465?v=4?s=100" width="100px;" alt="Itay Gabbay"/><br /><sub><b>Itay Gabbay</b></sub></a><br /><a href="#code-ItayGabbay" title="Code">💻</a> <a href="#doc-ItayGabbay" title="Documentation">📖</a> <a href="#ideas-ItayGabbay" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/matanper"><img src="https://avatars.githubusercontent.com/u/9868530?v=4?s=100" width="100px;" alt="matanper"/><br /><sub><b>matanper</b></sub></a><br /><a href="#doc-matanper" title="Documentation">📖</a> <a href="#ideas-matanper" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-matanper" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JKL98ISR"><img src="https://avatars.githubusercontent.com/u/26321553?v=4?s=100" width="100px;" alt="JKL98ISR"/><br /><sub><b>JKL98ISR</b></sub></a><br /><a href="#ideas-JKL98ISR" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-JKL98ISR" title="Code">💻</a> <a href="#doc-JKL98ISR" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yromanyshyn"><img src="https://avatars.githubusercontent.com/u/71635444?v=4?s=100" width="100px;" alt="Yurii Romanyshyn"/><br /><sub><b>Yurii Romanyshyn</b></sub></a><br /><a href="#ideas-yromanyshyn" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-yromanyshyn" title="Code">💻</a> <a href="#doc-yromanyshyn" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/noamzbr"><img src="https://avatars.githubusercontent.com/u/17730502?v=4?s=100" width="100px;" alt="Noam Bressler"/><br /><sub><b>Noam Bressler</b></sub></a><br /><a href="#code-noamzbr" title="Code">💻</a> <a href="#doc-noamzbr" title="Documentation">📖</a> <a href="#ideas-noamzbr" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nirhutnik"><img src="https://avatars.githubusercontent.com/u/92314933?v=4?s=100" width="100px;" alt="Nir Hutnik"/><br /><sub><b>Nir Hutnik</b></sub></a><br /><a href="#code-nirhutnik" title="Code">💻</a> <a href="#doc-nirhutnik" title="Documentation">📖</a> <a href="#ideas-nirhutnik" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Nadav-Barak"><img src="https://avatars.githubusercontent.com/u/67195469?v=4?s=100" width="100px;" alt="Nadav-Barak"/><br /><sub><b>Nadav-Barak</b></sub></a><br /><a href="#code-Nadav-Barak" title="Code">💻</a> <a href="#doc-Nadav-Barak" title="Documentation">📖</a> <a href="#ideas-Nadav-Barak" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TheSolY"><img src="https://avatars.githubusercontent.com/u/99395146?v=4?s=100" width="100px;" alt="Sol"/><br /><sub><b>Sol</b></sub></a><br /><a href="#code-TheSolY" title="Code">💻</a> <a href="#doc-TheSolY" title="Documentation">📖</a> <a href="#ideas-TheSolY" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.linkedin.com/in/dan-arlowski"><img src="https://avatars.githubusercontent.com/u/59116108?v=4?s=100" width="100px;" alt="DanArlowski"/><br /><sub><b>DanArlowski</b></sub></a><br /><a href="#code-DanArlowski" title="Code">💻</a> <a href="#infra-DanArlowski" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/benisraeldan"><img src="https://avatars.githubusercontent.com/u/42312361?v=4?s=100" width="100px;" alt="DBI"/><br /><sub><b>DBI</b></sub></a><br /><a href="#code-benisraeldan" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/OrlyShmorly"><img src="https://avatars.githubusercontent.com/u/110338263?v=4?s=100" width="100px;" alt="OrlyShmorly"/><br /><sub><b>OrlyShmorly</b></sub></a><br /><a href="#design-OrlyShmorly" title="Design">🎨</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shir22"><img src="https://avatars.githubusercontent.com/u/33841818?v=4?s=100" width="100px;" alt="shir22"/><br /><sub><b>shir22</b></sub></a><br /><a href="#ideas-shir22" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-shir22" title="Documentation">📖</a> <a href="#talk-shir22" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yaronzo1"><img src="https://avatars.githubusercontent.com/u/107114284?v=4?s=100" width="100px;" alt="yaronzo1"/><br /><sub><b>yaronzo1</b></sub></a><br /><a href="#ideas-yaronzo1" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-yaronzo1" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ptannor"><img src="https://avatars.githubusercontent.com/u/34207422?v=4?s=100" width="100px;" alt="ptannor"/><br /><sub><b>ptannor</b></sub></a><br /><a href="#ideas-ptannor" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-ptannor" title="Content">🖋</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/avitzd"><img src="https://avatars.githubusercontent.com/u/84308273?v=4?s=100" width="100px;" alt="avitzd"/><br /><sub><b>avitzd</b></sub></a><br /><a href="#eventOrganizing-avitzd" title="Event Organizing">📋</a> <a href="#video-avitzd" title="Videos">📹</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DanBasson"><img src="https://avatars.githubusercontent.com/u/46203939?v=4?s=100" width="100px;" alt="DanBasson"/><br /><sub><b>DanBasson</b></sub></a><br /><a href="#doc-DanBasson" title="Documentation">📖</a> <a href="#bug-DanBasson" title="Bug reports">🐛</a> <a href="#example-DanBasson" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kishore-s-15"><img src="https://avatars.githubusercontent.com/u/56688194?v=4?s=100" width="100px;" alt="S.Kishore"/><br /><sub><b>S.Kishore</b></sub></a><br /><a href="#code-Kishore-s-15" title="Code">💻</a> <a href="#doc-Kishore-s-15" title="Documentation">📖</a> <a href="#bug-Kishore-s-15" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.shaypalachy.com/"><img src="https://avatars.githubusercontent.com/u/917954?v=4?s=100" width="100px;" alt="Shay Palachy-Affek"/><br /><sub><b>Shay Palachy-Affek</b></sub></a><br /><a href="#data-Shaypal5" title="Data">🔣</a> <a href="#example-Shaypal5" title="Examples">💡</a> <a href="#userTesting-Shaypal5" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cemalgurpinar"><img src="https://avatars.githubusercontent.com/u/36713268?v=4?s=100" width="100px;" alt="Cemal GURPINAR"/><br /><sub><b>Cemal GURPINAR</b></sub></a><br /><a href="#doc-cemalgurpinar" title="Documentation">📖</a> <a href="#bug-cemalgurpinar" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/daavoo"><img src="https://avatars.githubusercontent.com/u/12677733?v=4?s=100" width="100px;" alt="David de la Iglesia Castro"/><br /><sub><b>David de la Iglesia Castro</b></sub></a><br /><a href="#code-daavoo" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Tak"><img src="https://avatars.githubusercontent.com/u/142250?v=4?s=100" width="100px;" alt="Levi Bard"/><br /><sub><b>Levi Bard</b></sub></a><br /><a href="#doc-Tak" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/julienschuermans"><img src="https://avatars.githubusercontent.com/u/14927054?v=4?s=100" width="100px;" alt="Julien Schuermans"/><br /><sub><b>Julien Schuermans</b></sub></a><br /><a href="#bug-julienschuermans" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.nirbenzvi.com"><img src="https://avatars.githubusercontent.com/u/4930255?v=4?s=100" width="100px;" alt="Nir Ben-Zvi"/><br /><sub><b>Nir Ben-Zvi</b></sub></a><br /><a href="#code-nirbenz" title="Code">💻</a> <a href="#ideas-nirbenz" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ashtavakra.org"><img src="https://avatars.githubusercontent.com/u/322451?v=4?s=100" width="100px;" alt="Shiv Shankar Dayal"/><br /><sub><b>Shiv Shankar Dayal</b></sub></a><br /><a href="#infra-shivshankardayal" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RonItay"><img src="https://avatars.githubusercontent.com/u/33497483?v=4?s=100" width="100px;" alt="RonItay"/><br /><sub><b>RonItay</b></sub></a><br /><a href="#bug-RonItay" title="Bug reports">🐛</a> <a href="#code-RonItay" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jeroen.vangoey.be"><img src="https://avatars.githubusercontent.com/u/59344?v=4?s=100" width="100px;" alt="Jeroen Van Goey"/><br /><sub><b>Jeroen Van Goey</b></sub></a><br /><a href="#bug-BioGeek" title="Bug reports">🐛</a> <a href="#doc-BioGeek" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://about.me/ido.weiss"><img src="https://avatars.githubusercontent.com/u/10072365?v=4?s=100" width="100px;" alt="idow09"/><br /><sub><b>idow09</b></sub></a><br /><a href="#bug-idow09" title="Bug reports">🐛</a> <a href="#example-idow09" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://bandism.net/"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt="Ikko Ashimine"/><br /><sub><b>Ikko Ashimine</b></sub></a><br /><a href="#doc-eltociear" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jhwohlgemuth"><img src="https://avatars.githubusercontent.com/u/6383605?v=4?s=100" width="100px;" alt="Jason Wohlgemuth"/><br /><sub><b>Jason Wohlgemuth</b></sub></a><br /><a href="#doc-jhwohlgemuth" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://lokin.dev"><img src="https://avatars.githubusercontent.com/u/34796341?v=4?s=100" width="100px;" alt="Lokin Sethia"/><br /><sub><b>Lokin Sethia</b></sub></a><br /><a href="#code-alphabetagamer" title="Code">💻</a> <a href="#bug-alphabetagamer" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.ingomarquart.de"><img src="https://avatars.githubusercontent.com/u/102803372?v=4?s=100" width="100px;" alt="Ingo Marquart"/><br /><sub><b>Ingo Marquart</b></sub></a><br /><a href="#code-IngoStatworx" title="Code">💻</a> <a href="#bug-IngoStatworx" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/osw282"><img src="https://avatars.githubusercontent.com/u/25309418?v=4?s=100" width="100px;" alt="Oscar"/><br /><sub><b>Oscar</b></sub></a><br /><a href="#code-osw282" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rcwoolston"><img src="https://avatars.githubusercontent.com/u/5957841?v=4?s=100" width="100px;" alt="Richard W"/><br /><sub><b>Richard W</b></sub></a><br /><a href="#code-rcwoolston" title="Code">💻</a> <a href="#doc-rcwoolston" title="Documentation">📖</a> <a href="#ideas-rcwoolston" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bgalvao"><img src="https://avatars.githubusercontent.com/u/17158288?v=4?s=100" width="100px;" alt="Bernardo"/><br /><sub><b>Bernardo</b></sub></a><br /><a href="#code-bgalvao" title="Code">💻</a> <a href="#doc-bgalvao" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://olivierbinette.github.io/"><img src="https://avatars.githubusercontent.com/u/784901?v=4?s=100" width="100px;" alt="Olivier Binette"/><br /><sub><b>Olivier Binette</b></sub></a><br /><a href="#code-OlivierBinette" title="Code">💻</a> <a href="#doc-OlivierBinette" title="Documentation">📖</a> <a href="#ideas-OlivierBinette" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chendingyan"><img src="https://avatars.githubusercontent.com/u/16874978?v=4?s=100" width="100px;" alt="陈鼎彦"/><br /><sub><b>陈鼎彦</b></sub></a><br /><a href="#bug-chendingyan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.k-lab.tk/"><img src="https://avatars.githubusercontent.com/u/16821717?v=4?s=100" width="100px;" alt="Andres Vargas"/><br /><sub><b>Andres Vargas</b></sub></a><br /><a href="#doc-vargasa" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MichaelMarien"><img src="https://avatars.githubusercontent.com/u/13829139?v=4?s=100" width="100px;" alt="Michael Marien"/><br /><sub><b>Michael Marien</b></sub></a><br /><a href="#doc-MichaelMarien" title="Documentation">📖</a> <a href="#bug-MichaelMarien" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mglowacki100"><img src="https://avatars.githubusercontent.com/u/6077538?v=4?s=100" width="100px;" alt="OrdoAbChao"/><br /><sub><b>OrdoAbChao</b></sub></a><br /><a href="#code-mglowacki100" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thewchan"><img src="https://avatars.githubusercontent.com/u/49702524?v=4?s=100" width="100px;" alt="Matt Chan"/><br /><sub><b>Matt Chan</b></sub></a><br /><a href="#code-thewchan" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hjain5164"><img src="https://avatars.githubusercontent.com/u/20479605?v=4?s=100" width="100px;" alt="Harsh Jain"/><br /><sub><b>Harsh Jain</b></sub></a><br /><a href="#code-hjain5164" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org)
specification. Contributions of any kind are welcome!


<!---

#############################
#############################
#############################
#############################
#############################

## 🔢 Tabular, 🖼️ Computer Vision & 🔤 NLP Support 

**This README refers to the Tabular version** of deepchecks.

- Check out the [Deepchecks for Computer Vision & Images subpackage](deepchecks/vision) for more details about deepchecks for CV, currently in *beta release*.
- Check out the [Deepchecks for NLP subpackage](deepchecks/nlp) for more details about deepchecks for NLP, currently in *beta release*.


## 💻 Installation


### Using pip

```bash
pip install deepchecks -U --user
```


> Note: Vision & NLP Install
>
> To install deepchecks together with the **Computer Vision Submodule** that 
> is currently in *beta release*, replace 
> ``deepchecks`` with ``"deepchecks[vision]"`` as follows:   
> ```bash
> pip install "deepchecks[vision]" -U --user
> ```
>  
> To install deepchecks together with the **NLP Submodule** that 
> is currently in *beta release*, replace 
> ``deepchecks`` with ``"deepchecks[nlp]"`` as follows:   
> ```bash
> pip install "deepchecks[nlp]" -U --user
> ```
>  
   
### Using conda

```bash
conda install -c conda-forge deepchecks
```



## ⏩ Try it Out!

### 🏃‍♀️ See It in Action

Head over to one of our following quickstart tutorials, and have deepchecks running on your environment in less than 5 min:

- [Train-Test Validation Quickstart (loans data)](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/plot_quick_train_test_validation.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out)
- [Data Integrity Quickstart (avocado sales data)](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/plot_quick_data_integrity.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out)
- [Model Evaluation Quickstart (wine quality data)](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/plot_quickstart_in_5_minutes.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out)

> **Recommended - download the code and run it locally** on the built-in dataset and (optional) model, or **replace them with your own**.


### 🚀 See Our Checks Demo

Play with some of the existing checks in our [Interactive Checks Demo](
   https://checks-demo.deepchecks.com/?check=No+check+selected&utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out), 
and see how they work on various datasets with custom corruptions injected.


## 📊 Usage Examples

### Running a Suite

A [Suite](#suite) runs a collection of [Checks](#check) with optional
[Conditions](#condition) added to them.

Example for running a suite on given
[datasets](https://docs.deepchecks.com/stable/user-guide/tabular/dataset_object.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=running_a_suite)
and with a [supported
model](https://docs.deepchecks.com/stable/user-guide/supported_models.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=running_a_suite):

```python
from deepchecks.tabular.suites import model_evaluation
suite = model_evaluation()
result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
result.save_as_html() # replace this with result.show() or result.show_in_window() to see results inline or in window
```

Which will result in a report that looks like this:


<p align="center">
   <img src="docs/source/_static/images/general/model_evaluation_suite.gif" width="800">
</p>


Note:

- Results can be [displayed](https://docs.deepchecks.com/stable/user-guide/general/showing_results.html) in various manners, or [exported](https://docs.deepchecks.com/stable/user-guide/general/export_save_results.html) to an html report, saved as JSON, or integrated with other tools (e.g. wandb).
- Other suites that run only on the data (``data_integrity``, ``train_test_validation``) don't require a model as part of the input.

See the [full code tutorials
here](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/index.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=try_it_out).


In the following section you can see an example of how the output of a single check without a condition may look.

### Running a Check

To run a specific single check, all you need to do is import it and then
to run it with the required (check-dependent) input parameters. More
details about the existing checks and the parameters they can receive
can be found in our [API
Reference](https://docs.deepchecks.com/stable/api/index.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=running_a_check).

```python
from deepchecks.tabular.checks import FeatureDrift
import pandas as pd

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
# Initialize and run desired check
FeatureDrift().run(train_df, test_df)
```

Will produce output of the type:

>   <h4>Train Test Drift</h4>
>  <p>The Drift score is a measure for the difference between two distributions,
>   in this check - the test and train distributions. <br>
>   The check shows the drift score and distributions for the features,
>   sorted by feature importance and showing only the top 5 features, according to feature importance.
>   If available, the plot titles also show the feature importance (FI) rank.</p>
>   <p align="left">
>      <img src="docs/source/_static/images/general/train-test-drift-output.png">
>   </p>


## 🙋🏼  When Should You Use Deepchecks?

While you’re in the research phase, and want to validate your data, find potential methodological problems, 
and/or validate your model and evaluate it.


<p align="center">
   <img src="/docs/source/_static/images/general/pipeline_when_to_validate.svg">
</p>


See more about typical usage scenarios and the built-in suites in the [docs](
   https://docs.deepchecks.com/stable/getting-started/welcome.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=what_do_you_need_in_order_to_start_validating#when-should-you-use-deepchecks).


## 🗝️ Key Concepts

### Check

Each check enables you to inspect a specific aspect of your data and
models. They are the basic building block of the deepchecks package,
covering all kinds of common issues, such as:

- Weak Segments Performance
- Feature Drift
- Date Train Test Leakage Overlap
- Conflicting Labels 

and [many more
checks](https://docs.deepchecks.com/stable/tabular/index.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=key_concepts__check).


Each check can have two types of
results:

1. A visual result meant for display (e.g. a figure or a table).
2. A return value that can be used for validating the expected check
   results (validations are typically done by adding a "condition" to
   the check, as explained below).

### Condition

A condition is a function that can be added to a Check, which returns a
pass ✓, fail ✖ or warning ! result, intended for validating the Check's
return value. An example for adding a condition would be:

```python
from deepchecks.tabular.checks import BoostingOverfit
BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(threshold=0.05)
```

which will return a check failure when running it if there is a difference of
more than 5% between the best score achieved on the test set during the boosting
iterations and the score achieved in the last iteration (the model's "original" score
on the test set).

### Suite

An ordered collection of checks, that can have conditions added to them.
The Suite enables displaying a concluding report for all of the Checks
that ran.

See the list of [predefined existing suites](deepchecks/tabular/suites)
for tabular data to learn more about the suites you can work with
directly and also to see a code example demonstrating how to build your
own custom suite.

The existing suites include default conditions added for most of the checks.
You can edit the preconfigured suites or build a suite of your own with a collection
of checks and optional conditions.


<p align="center">
   <img src="/docs/source/_static/images/general/diagram.svg">
</p>


## 🤔 What Do You Need in Order to Start Validating?

### Environment

- The deepchecks package installed
- JupyterLab or Jupyter Notebook or any Python IDE


### Data / Model 

Depending on your phase and what you wish to validate, you'll need a
subset of the following:

-  Raw data (before pre-processing such as OHE, string processing,
   etc.), with optional labels
-  The model's training data with labels
-  Test data (which the model isn't exposed to) with labels
-  A [supported
    model](https://docs.deepchecks.com/stable/user-guide/supported_models.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=running_a_suite) (e.g. scikit-learn models, XGBoost, any model implementing the ``predict`` method in the required format)


### Supported Data Types

The package currently supports tabular data and is in:
- *beta release* for the [Computer Vision subpackage](deepchecks/vision).
- *beta release* for the [NLP subpackage](deepchecks/nlp).

--->