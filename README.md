# m2cgen

[![GitHub Actions Status](https://github.com/schen18/m2cgen/workflows/GitHub%20Actions/badge.svg?branch=master)](https://github.com/schen18/m2cgen/actions)
[![License: MIT](https://img.shields.io/github/license/schen18/m2cgen.svg)](https://github.com/schen18/m2cgen/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/m2cgen-refresh.svg?logo=python&logoColor=white)](https://pypi.org/project/m2cgen-refresh)
[![PyPI Version](https://img.shields.io/pypi/v/m2cgen-refresh.svg?logo=pypi&logoColor=white)](https://pypi.org/project/m2cgen-refresh)
[![Downloads](https://pepy.tech/badge/m2cgen-refresh)](https://pepy.tech/project/m2cgen-refresh)

**m2cgen** (Model 2 Code Generator) - is a lightweight library which provides an easy way to transpile trained statistical models into a native code (Python, C, Java, Go, JavaScript, Visual Basic, C#, PowerShell, R, PHP, Dart, Haskell, Ruby, F#, Rust, Elixir).

* [Installation](#installation)
* [Development](#development)
* [Supported Languages](#supported-languages)
* [Supported Models](#supported-models)
* [Classification Output](#classification-output)
* [Usage](#usage)
* [CLI](#cli)
* [FAQ](#faq)

## Installation
Supported Python version is >= **3.12**.
```
pip install m2cgen
```

## Development
Make sure the following command runs successfully before submitting a PR:
```
make pre-pr
```
Alternatively you can run the Docker version of the same command:
```
make docker-build docker-pre-pr
```

## Supported Languages

- C#
- Java
- JavaScript
- PHP
- PowerShell
- Python
- Rust
- SQL (DuckDB macros, MySQL/MariaDB stored functions, PostgreSQL functions)
- Visual Basic (VBA-compatible)

## Supported Models

|  | Classification | Regression |
| --- | --- | --- |
| **Linear** | <ul><li>scikit-learn<ul><li>LinearDiscriminantAnalysis</li><li>LogisticRegression</li><li>LogisticRegressionCV</li><li>PassiveAggressiveClassifier</li><li>Perceptron</li><li>RidgeClassifier</li><li>RidgeClassifierCV</li><li>SGDClassifier</li></ul></li><li>scikit-learn (Naive Bayes)<ul><li>BernoulliNB</li><li>ComplementNB</li><li>GaussianNB</li><li>MultinomialNB</li></ul></li><li>wittgenstein<ul><li>IREP</li><li>RIPPER</li></ul></li></ul> | <ul><li>scikit-learn<ul><li>ARDRegression</li><li>BayesianRidge</li><li>ElasticNet</li><li>ElasticNetCV</li><li>GammaRegressor</li><li>HuberRegressor</li><li>Lars</li><li>LarsCV</li><li>Lasso</li><li>LassoCV</li><li>LassoLars</li><li>LassoLarsCV</li><li>LassoLarsIC</li><li>LinearRegression</li><li>OrthogonalMatchingPursuit</li><li>OrthogonalMatchingPursuitCV</li><li>PassiveAggressiveRegressor</li><li>PoissonRegressor</li><li>RANSACRegressor(only supported regression estimators can be used as a base estimator)</li><li>Ridge</li><li>RidgeCV</li><li>SGDRegressor</li><li>TheilSenRegressor</li><li>TweedieRegressor</li></ul><li>StatsModels<ul><li>Generalized Least Squares (GLS)</li><li>Generalized Least Squares with AR Errors (GLSAR)</li><li>Generalized Linear Models (GLM)</li><li>Ordinary Least Squares (OLS)</li><li>[Gaussian] Process Regression Using Maximum Likelihood-based Estimation (ProcessMLE)</li>
<li>Logit (via discrete models)</li>
<li>MNLogit (via discrete models)</li>
<li>NegativeBinomial (via discrete models)</li>
<li>OrderedModel (logit distribution only)</li>
<li>Poisson (via discrete models)</li>
<li>RLM (robust linear model)</li><li>Quantile Regression (QuantReg)</li><li>Weighted Least Squares (WLS)</li></ul></ul> |
| **Anomaly detection** | <ul><li>scikit-learn<ul><li>IsolationForest</li></ul></li></ul> | <ul></ul> |
| **SVM** | <ul><li>scikit-learn<ul><li>LinearSVC</li><li>NuSVC</li><li>OneClassSVM</li><li>SVC</li></ul></li></ul> | <ul><li>scikit-learn<ul><li>LinearSVR</li><li>NuSVR</li><li>SVR</li></ul></li></ul> |
| **Clustering** | <ul><li>scikit-learn<ul><li>GaussianMixture</li><li>BayesianGaussianMixture</li></ul></li></ul> | <ul><li>scikit-learn<ul><li>BisectingKMeans</li><li>KMeans</li><li>MiniBatchKMeans</li></ul></li></ul> |
| **Tree** | <ul><li>DecisionTreeClassifier</li><li>ExtraTreeClassifier</li></ul> | <ul><li>DecisionTreeRegressor</li><li>ExtraTreeRegressor</li></ul> |
| **Random Forest** | <ul><li>ExtraTreesClassifier</li><li>LGBMClassifier(rf booster only)</li><li>RandomForestClassifier</li><li>XGBRFClassifier</li></ul> | <ul><li>ExtraTreesRegressor</li><li>LGBMRegressor(rf booster only)</li><li>RandomForestRegressor</li><li>XGBRFRegressor</li></ul> |
| **Boosting** | <ul><li>LGBMClassifier(gbdt/dart/goss booster only)</li><li>XGBClassifier(gbtree(including boosted forests)/gblinear booster only)</li><ul> | <ul><li>LGBMRegressor(gbdt/dart/goss booster only)</li><li>XGBRegressor(gbtree(including boosted forests)/gblinear booster only)</li></ul> |

You can find versions of packages with which compatibility is guaranteed by CI tests [here](https://github.com/schen18/m2cgen/blob/master/requirements-test.txt#L1).
Other versions can also be supported but they are untested.

## Classification Output
### Linear / Linear SVM / Kernel SVM
#### Binary
Scalar value; signed distance of the sample to the hyperplane for the second class.
#### Multiclass
Vector value; signed distance of the sample to the hyperplane per each class.
#### Comment
The output is consistent with the output of ```LinearClassifierMixin.decision_function``` (and of ```LinearDiscriminantAnalysis.decision_function```).

### SVM
#### Outlier detection
Scalar value; signed distance of the sample to the separating hyperplane: positive for an inlier and negative for an outlier.
#### Binary
Scalar value; signed distance of the sample to the hyperplane for the second class.
#### Multiclass
Vector value; one-vs-one score for each class, shape (n_samples, n_classes * (n_classes-1) / 2).
#### Comment
The output is consistent with the output of ```BaseSVC.decision_function``` when the `decision_function_shape` is set to `ovo`.

### Naive Bayes
Vector value; class probabilities (`predict_proba`). The class label (`predict`, the argmax) is the index of the largest value.

### Isolation forest
Scalar value; anomaly score (`decision_function`). Negative values indicate anomalies (as classified by `predict`).

### Statsmodels discrete-choice models
#### Logit
Scalar value; probability of the positive class (`predict`).
#### MNLogit / OrderedModel
Vector value; class probabilities (`predict`).
#### Poisson / NegativeBinomial
Scalar value; expected count (`predict`).
#### Comment
Probit models (and OrderedModel with the probit distribution) are not supported: they require the erf function which is not available in all target languages.

### Rule-based (wittgenstein)
#### Binary
Vector value; class probabilities (`predict_proba`): the weighted average of the smoothed class frequencies of all covering rules, or the ruleset default for samples not covered by any rule. Only numeric (including library-discretized) conditions are supported.

### Clustering
#### KMeans / MiniBatchKMeans / BisectingKMeans
Vector value; Euclidean distance to each cluster center.
The cluster assignment (`predict`) is the index of the smallest value.
#### GaussianMixture / BayesianGaussianMixture
Vector value; cluster responsibilities.
The cluster assignment (`predict`) is the index of the largest value.
#### Comment
The output is consistent with the output of the `transform` method of `KMeans` / `MiniBatchKMeans` / `BisectingKMeans` and of the `predict_proba` method of `GaussianMixture` / `BayesianGaussianMixture`.

### Tree / Random Forest / Boosting
#### Binary
Vector value; class probabilities.
#### Multiclass
Vector value; class probabilities.
#### Comment
The output is consistent with the output of the `predict_proba` method of `DecisionTreeClassifier` / `ExtraTreeClassifier` / `ExtraTreesClassifier` / `RandomForestClassifier` / `XGBRFClassifier` / `XGBClassifier` / `LGBMClassifier`.

## Usage

SQL output is a function/macro taking one parameter per feature, so it can be
used directly with your own column names (no renaming needed):

```python
import pandas as pd
from sklearn import linear_model
import m2cgen as m2c

X = pd.DataFrame(..., columns=["age", "bmi", "bp"])
estimator = linear_model.LinearRegression().fit(X, y)

code = m2c.export_to_sql(estimator)  # dialect="duckdb" by default;
                                     # "mysql" and "postgres" are also supported
```

Generated DuckDB macro (feature names are picked up automatically from the
model when it was trained on a pandas DataFrame):

```sql
CREATE OR REPLACE MACRO score(age, bmi, bp) AS (
    SELECT 0.23E0 + age * 0.53E0 + bmi * 0.21E0 + bp * 0.11E0
);
```

Usage: `SELECT score(age, bmi, bp) FROM some_table`. Note that NULL inputs
propagate to `ELSE` branches in SQL, which differs from the NaN handling of
the other supported languages. Also note that DuckDB is a vectorized engine:
evaluating a macro over a table (many rows per query) is dramatically faster
than calling it on single rows one at a time.

Here's a simple example of how a linear model trained in Python environment can be represented in Java code:
```python
from sklearn.datasets import load_diabetes
from sklearn import linear_model
import m2cgen as m2c

X, y = load_diabetes(return_X_y=True)

estimator = linear_model.LinearRegression()
estimator.fit(X, y)

code = m2c.export_to_java(estimator)
```

Generated Java code:
```java
public class Model {
    public static double score(double[] input) {
        return ((((((((((152.1334841628965) + ((input[0]) * (-10.012197817470472))) + ((input[1]) * (-239.81908936565458))) + ((input[2]) * (519.8397867901342))) + ((input[3]) * (324.39042768937657))) + ((input[4]) * (-792.1841616283054))) + ((input[5]) * (476.74583782366153))) + ((input[6]) * (101.04457032134408))) + ((input[7]) * (177.06417623225025))) + ((input[8]) * (751.2793210873945))) + ((input[9]) * (67.62538639104406));
    }
}
```

**You can find more examples of generated code for different models/languages [here](https://github.com/schen18/m2cgen/tree/master/generated_code_examples).**

## CLI

`m2cgen` can be used as a CLI tool to generate code using serialized model objects (pickle protocol):
```
$ m2cgen <pickle_file> --language <language> [--indent <indent>] [--function_name <function_name>]
         [--class_name <class_name>] [--module_name <module_name>] [--package_name <package_name>]
         [--namespace <namespace>] [--recursion-limit <recursion_limit>]
```
Don't forget that for unpickling serialized model objects their classes must be defined in the top level of an importable module in the unpickling environment.

Piping is also supported:
```
$ cat <pickle_file> | m2cgen --language <language>
```

## FAQ
**Q: Generation fails with `RecursionError: maximum recursion depth exceeded` error.**

A: If this error occurs while generating code using an ensemble model, try to reduce the number of trained estimators within that model. Alternatively you can increase the maximum recursion depth with `sys.setrecursionlimit(<new_depth>)`.

**Q: Generation fails with `ImportError: No module named <module_name_here>` error while transpiling model from a serialized model object.**

A: This error indicates that pickle protocol cannot deserialize model object. For unpickling serialized model objects, it is required that their classes must be defined in the top level of an importable module in the unpickling environment. So installation of package which provided model's class definition should solve the problem.

**Q: Generated by m2cgen code provides different results for some inputs compared to original Python model from which the code were obtained.**

A: Tree ensembles of XGBoost and LightGBM compare the ``float32`` representations of input features against the ``float32`` tree thresholds, while the generated code compares the original ``float64`` values. For inputs whose values lie in the tiny gap between the two representations the code can therefore follow a different branch of a tree. This is most likely to happen with features that contain many repeated or rounded values. Also, some small differences can happen due to specific implementation of floating-point arithmetic in a target language.
