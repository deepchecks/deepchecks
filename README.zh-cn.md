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


<p align="center">
   &emsp;
   <a href="https://www.deepchecks.com/slack">加入&nbsp;Slack </a>
   &emsp; | &emsp; 
   <a href="https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=top_links">文件</a>
   &emsp; | &emsp; 
   <a href="https://deepchecks.com/blog/?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=top_links">博客</a>
   &emsp; | &emsp;  
   <a href="https://twitter.com/deepchecks">推特</a>
   &emsp;
</p>
   

<p align="center">
   <a href="https://deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=logo">
      <img src="docs/source/_static/images/general/deepchecks-logo-with-white-wide-back.png">
   </a>
</p>


[![GitHub
stars](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/deepchecks/deepchecks/stargazers/)
![build](https://github.com/deepchecks/deepchecks/actions/workflows/build.yml/badge.svg)
![pkgVersion](https://img.shields.io/pypi/v/deepchecks)
![pyVersions](https://img.shields.io/pypi/pyversions/deepchecks)
[![Maintainability](https://api.codeclimate.com/v1/badges/970b11794144139975fa/maintainability)](https://codeclimate.com/github/deepchecks/deepchecks/maintainability)
[![Coverage
Status](https://coveralls.io/repos/github/deepchecks/deepchecks/badge.svg?branch=main)](https://coveralls.io/github/deepchecks/deepchecks?branch=main)


<h1 align="center">
   测试并验证ML模型和数据
</h1>

<div align="center">
   
   [English](./README.md) | 简体中文
</div>

<p align="center">
   <a href="https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=checks_and_conditions_img">
   <img src="docs/source/_static/images/general/checks-and-conditions.png">
   </a>
</p>


## 🧐 Deepchecks是什么

Deepchecks是一个Python包，可以轻而易举地全面验证您的机器学习模型和数据。其中包括与各类问题相关的检查，例如模型性能、数据完整性、分布适配等。


## 🖼️ 计算机视觉和 🔢 表格支持

**本README是指** deepchecks的表格版本。

查看 [Deepchecks计算机视觉和图像包，](deepchecks/vision) 了解有关deepchecks计算机视觉的更多信息（目前为 *beta版本f*）。


## 💻 安装


### 使用pip

```bash
pip install deepchecks -U --user
```

> 注：计算机视觉包安装
>
> 若要与目前为 *beta版本的* **计算机视觉子模块** 一同安装 deepchecks，请将
> ``deepchecks`` 替换为 ``"deepchecks[vision]"``，如下所示:
>  
> ```bash
> pip install "deepchecks[vision]" -U --user
> ```
>  
   
### 使用conda

```bash
conda install -c conda-forge deepchecks
```


## ⏩ 试一下吧

### 🏃‍♀️ 看它如何发挥作用

前往我们以下快速入门教程之一，不用5分钟即可让deepchecks在您的环境中运行。

- [训练测试验证快速入门](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/plot_quick_train_test_validation.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=try_it_out)
- [数据完整性快速入门](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/plot_quick_data_integrity.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=try_it_out)
- [模型评估快速入门](
   https://docs.deepchecks.com/en/stable/user-guide/tabular/auto_quickstarts/plot_quickstart_in_5_minutes.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=try_it_out)

> **推荐 - 下载代码并** 在内置数据集和（可选）模型中本地运行，或 **将其替换为您自己的内容**。


### 🚀 查看我们的检查演示

在我们的 [交互式检查演示](
   https://checks-demo.deepchecks.com/?check=No+check+selected&utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=try_it_out), 
中运行一些现有检查，看其如何在注入自定义损坏的情况下，在各种数据集中工作。


## 📊 使用示例

### 运行套件

一个 [套件](#suite) 运行一组附有可选 [条件](#check) 的
[检查](#condition)。

在给定的
[数据集](https://docs.deepchecks.com/stable/user-guide/tabular/dataset_object.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=running_a_suite)
中，通过 [受支持的模型](https://docs.deepchecks.com/stable/user-guide/supported_models.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=running_a_suite) 运行套件的示例:

```python
from deepchecks.tabular.suites import model_evaluation
suite = model_evaluation()
result = suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=model)
result.save_as_html() # 将此替换为result.show() or result.show_in_window()，以在行内或窗口中查看结果
```

这将产生如下所示的报告：

<p align="center">
   <img src="docs/source/_static/images/general/model_evaluation_suite.gif" width="800">
</p>


注：

- 结果可以多种方式[显示](https://docs.deepchecks.com/stable/user-guide/general/showing_results.html), [导出](https://docs.deepchecks.com/stable/user-guide/general/export_save_results.html)到 html 报告，保存为 JSON，或与其他工具（例如 wandb）集成。
- 仅在数据 (``data_integrity``, ``train_test_validation``) 中运行的其它套件，无需将模型作为输入的一部分。

在此查看 [完整代码教程](
   https://docs.deepchecks.com/stable/user-guide/tabular/auto_quickstarts/index.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=try_it_out)。


在以下部分，您可看到一个示例，说明没有条件的单项检查的输出看上去会怎样。

### 运行检查

若要运行某个特定的单项检查，您需要做的只是将其导入，然后采用所需（依赖于检查）的输入参数运行即可。有关现有检查及其可以获得的参数的更多信息，请在我们的 [API参考](https://docs.deepchecks.com/stable/api/index.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=running_a_check) 中查找。

```python
from deepchecks.tabular.checks import TrainTestFeatureDrift
import pandas as pd

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
# 初始化并运行所需检查
TrainTestFeatureDrift().run(train_df, test_df)
```

将会产生以下类型的输出：

>   <h4>Train Test Drift</h4>
>  <p>The Drift score is a measure for the difference between two distributions,
>   in this check - the test and train distributions. <br>
>   The check shows the drift score and distributions for the features,
>   sorted by feature importance and showing only the top 5 features, according to feature importance.
>   If available, the plot titles also show the feature importance (FI) rank.</p>
>   <p align="left">
>      <img src="docs/source/_static/images/general/train-test-drift-output.png">
>   </p>


## 🙋🏼 何时应该使用 Deepchecks？

您在处于研究阶段，想要验证数据、找出潜在方法问题和/或验证并评估您的模型时。


<p align="center">
   <img src="/docs/source/_static/images/general/pipeline_when_to_validate.svg">
</p>


在 [docs](
   https://docs.deepchecks.com/stable/getting-started/welcome.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=what_do_you_need_in_order_to_start_validating#when-should-you-use-deepchecks) 中查看有关典型使用场景和内置套件的更多信息。


## 🗝️ 主要概念

### 检查

每项检查均可使您能够检查数据和模型的某个特定方面。它们是deepchecks包的基本构件，涵盖各种常见问题，例如：
 
- Weak Segments Performance
- Train Test Feature Drift
- Date Train Test Leakage Overlap
- Conflicting Labels

以及 [多项其它检查](https://docs.deepchecks.com/stable/checks_gallery/tabular.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=key_concepts__check)。


每项检查均会有两种结果：

1. 用于显示的视觉结果（例如图标或表格）。
2. 可用于验证预期检查结果的返回值（通常情况下，将“条件”添加到检查中，从而进行验证，如下所述）。

### 条件

条件是可以添加到检查的函数，它会返回一个pass ✓、fail ✖ 或warning ! 结果，用于验证检查的返回值。添加条件的示例如下：

```python
from deepchecks.tabular.checks import BoostingOverfit
BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(threshold=0.05)
```

如果在提升迭代期获得的测试集最佳分数与最后一次迭代中获得的分数（模型的测试集“原始”分数）之间存在5%以上的差异，则将在运行时返回检查失败。

### 套件

检查的有序集合，可以添加条件。套件可以显示所有已运行检查的结论报告。

参阅表格数据的 [预定义现有套件](deepchecks/tabular/suites) 
列表，以了解有关您可直接用其工作的套件的更多信息，查看演示如何构建您自己的自定义套件的代码示例。

现有套件包括为大多数检查添加的默认条件。您可编辑预配置套件，也可采用一组检查和可选条件构建您自己的套件。


<p align="center">
   <img src="/docs/source/_static/images/general/diagram.svg">
</p>


## 🤔 您需要些什么才能开始验证？

### 环境

- deepchecks包已安装
- JupyterLab或Jupyter Notebook或任何Python IDE


### 数据 / 模型

根据您所处的阶段以及您希望验证的内容，您将需要以下内容的子集：

-  原始数据（预处理之前，例如OHE、字符串处理等），附有可选标签
-  模型训练数据，附有标签
-  测试数据（模型未触及），附有标签
-  [受支持的模型](https://docs.deepchecks.com/stable/user-guide/supported_models.html?utm_source=github.com&utm_medium=referral&utm_campaign=readme_cn&utm_content=running_a_suite) (例如scikit-learn模型、XGBoost、任何以所需格式实现预测方法的模型）


### 受支持的数据类型

deepchecks包目前支持表格数据，现为  *beta版本*，用于 [计算机视觉子包](deepchecks/vision)。


## 📖 文件

-   [https://docs.deepchecks.com/](https://docs.deepchecks.com/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation) -   HTML 文件（最新版本）
-   [https://docs.deepchecks.com/dev](https://docs.deepchecks.com/dev/?utm_source=github.com&utm_medium=referral&utm_campaign=readme&utm_content=documentation) -   HTML 文件（开发版 - 主分支）


## 👭 社区

-   加入我们的 [Slack社区](https://www.deepchecks.com/slack)，与维护人员建立联系，关注用户和有趣的讨论
-   发布 [Github问题](https://github.com/deepchecks/deepchecks/issues)，以提出改进建议、引出问题或分享反馈。