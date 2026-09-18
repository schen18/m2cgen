from m2cgen.assemblers.boosting import (
    LightGBMModelAssembler,
    XGBoostLinearModelAssembler,
    XGBoostModelAssemblerSelector,
    XGBoostTreeModelAssembler
)
from m2cgen.assemblers.clustering import GaussianMixtureModelAssembler, KMeansModelAssembler
from m2cgen.assemblers.ensemble import RandomForestModelAssembler
from m2cgen.assemblers.isolation_forest import IsolationForestModelAssembler
from m2cgen.assemblers.linear import (
    ProcessMLEModelAssembler,
    SklearnGLMModelAssembler,
    SklearnLinearModelAssembler,
    StatsmodelsGLMModelAssembler,
    StatsmodelsLinearModelAssembler,
    StatsmodelsModelAssemblerSelector
)
from m2cgen.assemblers.meta import RANSACModelAssembler
from m2cgen.assemblers.naive_bayes import (
    BernoulliNaiveBayesModelAssembler,
    ComplementNaiveBayesModelAssembler,
    GaussianNaiveBayesModelAssembler,
    MultinomialNaiveBayesModelAssembler
)
from m2cgen.assemblers.ripper import WittgensteinRuleModelAssembler
from m2cgen.assemblers.statsmodels_discrete import (
    StatsmodelsBinaryModelAssembler,
    StatsmodelsCountModelAssembler,
    StatsmodelsMultinomialModelAssembler,
    StatsmodelsOrderedModelAssembler
)
from m2cgen.assemblers.svm import SklearnSVMModelAssembler
from m2cgen.assemblers.tree import TreeModelAssembler

__all__ = [
    SklearnLinearModelAssembler,
    StatsmodelsLinearModelAssembler,
    ProcessMLEModelAssembler,
    RANSACModelAssembler,
    TreeModelAssembler,
    RandomForestModelAssembler,
    XGBoostModelAssemblerSelector,
    XGBoostTreeModelAssembler,
    XGBoostLinearModelAssembler,
    LightGBMModelAssembler,
    SklearnSVMModelAssembler,
    StatsmodelsGLMModelAssembler,
    StatsmodelsModelAssemblerSelector,
    SklearnGLMModelAssembler,
    KMeansModelAssembler,
    GaussianMixtureModelAssembler,
    IsolationForestModelAssembler,
    GaussianNaiveBayesModelAssembler,
    MultinomialNaiveBayesModelAssembler,
    ComplementNaiveBayesModelAssembler,
    BernoulliNaiveBayesModelAssembler,
    StatsmodelsBinaryModelAssembler,
    StatsmodelsMultinomialModelAssembler,
    StatsmodelsCountModelAssembler,
    StatsmodelsOrderedModelAssembler,
    WittgensteinRuleModelAssembler,
]


SUPPORTED_MODELS = {
    # LightGBM
    "lightgbm_LGBMClassifier": LightGBMModelAssembler,
    "lightgbm_LGBMRegressor": LightGBMModelAssembler,

    # XGBoost
    "xgboost_XGBClassifier": XGBoostModelAssemblerSelector,
    "xgboost_XGBRFClassifier": XGBoostModelAssemblerSelector,
    "xgboost_XGBRegressor": XGBoostModelAssemblerSelector,
    "xgboost_XGBRFRegressor": XGBoostModelAssemblerSelector,

    # Sklearn SVM
    "sklearn_LinearSVC": SklearnLinearModelAssembler,
    "sklearn_LinearSVR": SklearnLinearModelAssembler,
    "sklearn_NuSVC": SklearnSVMModelAssembler,
    "sklearn_NuSVR": SklearnSVMModelAssembler,
    "sklearn_OneClassSVM": SklearnSVMModelAssembler,
    "sklearn_SVC": SklearnSVMModelAssembler,
    "sklearn_SVR": SklearnSVMModelAssembler,

    # Sklearn Linear Regressors
    "sklearn_ARDRegression": SklearnLinearModelAssembler,
    "sklearn_BayesianRidge": SklearnLinearModelAssembler,
    "sklearn_ElasticNet": SklearnLinearModelAssembler,
    "sklearn_ElasticNetCV": SklearnLinearModelAssembler,
    "sklearn_GammaRegressor": SklearnGLMModelAssembler,
    "sklearn_HuberRegressor": SklearnLinearModelAssembler,
    "sklearn_Lars": SklearnLinearModelAssembler,
    "sklearn_LarsCV": SklearnLinearModelAssembler,
    "sklearn_Lasso": SklearnLinearModelAssembler,
    "sklearn_LassoCV": SklearnLinearModelAssembler,
    "sklearn_LassoLars": SklearnLinearModelAssembler,
    "sklearn_LassoLarsCV": SklearnLinearModelAssembler,
    "sklearn_LassoLarsIC": SklearnLinearModelAssembler,
    "sklearn_LinearRegression": SklearnLinearModelAssembler,
    "sklearn_OrthogonalMatchingPursuit": SklearnLinearModelAssembler,
    "sklearn_OrthogonalMatchingPursuitCV": SklearnLinearModelAssembler,
    "sklearn_PassiveAggressiveRegressor": SklearnLinearModelAssembler,
    "sklearn_PoissonRegressor": SklearnGLMModelAssembler,
    "sklearn_RANSACRegressor": RANSACModelAssembler,
    "sklearn_Ridge": SklearnLinearModelAssembler,
    "sklearn_RidgeCV": SklearnLinearModelAssembler,
    "sklearn_SGDRegressor": SklearnLinearModelAssembler,
    "sklearn_TheilSenRegressor": SklearnLinearModelAssembler,
    "sklearn_TweedieRegressor": SklearnGLMModelAssembler,

    # Statsmodels Linear Regressors
    "statsmodels_GLMResultsWrapper": StatsmodelsGLMModelAssembler,
    "statsmodels_ProcessMLEResults": ProcessMLEModelAssembler,
    "statsmodels_RegressionResultsWrapper": StatsmodelsLinearModelAssembler,
    "statsmodels_RegularizedResultsWrapper": StatsmodelsModelAssemblerSelector,

    # Sklearn Linear Classifiers
    "sklearn_LinearDiscriminantAnalysis": SklearnLinearModelAssembler,
    "sklearn_LogisticRegression": SklearnLinearModelAssembler,
    "sklearn_LogisticRegressionCV": SklearnLinearModelAssembler,
    "sklearn_PassiveAggressiveClassifier": SklearnLinearModelAssembler,
    "sklearn_Perceptron": SklearnLinearModelAssembler,
    "sklearn_RidgeClassifier": SklearnLinearModelAssembler,
    "sklearn_RidgeClassifierCV": SklearnLinearModelAssembler,
    "sklearn_SGDClassifier": SklearnLinearModelAssembler,

    # Decision trees
    "sklearn_DecisionTreeClassifier": TreeModelAssembler,
    "sklearn_DecisionTreeRegressor": TreeModelAssembler,
    "sklearn_ExtraTreeClassifier": TreeModelAssembler,
    "sklearn_ExtraTreeRegressor": TreeModelAssembler,

    # Naive Bayes
    "sklearn_GaussianNB": GaussianNaiveBayesModelAssembler,
    "sklearn_MultinomialNB": MultinomialNaiveBayesModelAssembler,
    "sklearn_ComplementNB": ComplementNaiveBayesModelAssembler,
    "sklearn_BernoulliNB": BernoulliNaiveBayesModelAssembler,

    # Isolation forest
    "sklearn_IsolationForest": IsolationForestModelAssembler,

    # Statsmodels discrete-choice and robust models
    "statsmodels_BinaryResultsWrapper": StatsmodelsBinaryModelAssembler,
    "statsmodels_MultinomialResultsWrapper": StatsmodelsMultinomialModelAssembler,
    "statsmodels_PoissonResultsWrapper": StatsmodelsCountModelAssembler,
    "statsmodels_NegativeBinomialResultsWrapper": StatsmodelsCountModelAssembler,
    "statsmodels_OrderedResultsWrapper": StatsmodelsOrderedModelAssembler,
    "statsmodels_RLMResultsWrapper": StatsmodelsLinearModelAssembler,

    # Wittgenstein rule-based classifiers
    "wittgenstein_RIPPER": WittgensteinRuleModelAssembler,
    "wittgenstein_IREP": WittgensteinRuleModelAssembler,

    # Clustering
    "sklearn_KMeans": KMeansModelAssembler,
    "sklearn_MiniBatchKMeans": KMeansModelAssembler,
    "sklearn_BisectingKMeans": KMeansModelAssembler,
    "sklearn_GaussianMixture": GaussianMixtureModelAssembler,
    "sklearn_BayesianGaussianMixture": GaussianMixtureModelAssembler,

    # Ensembles
    "sklearn_ExtraTreesClassifier": RandomForestModelAssembler,
    "sklearn_ExtraTreesRegressor": RandomForestModelAssembler,
    "sklearn_RandomForestClassifier": RandomForestModelAssembler,
    "sklearn_RandomForestRegressor": RandomForestModelAssembler,
}


def _get_full_model_name(model):
    type_name = type(model)
    return f"{type_name.__module__.split('.')[0]}_{type_name.__name__}"


def get_assembler_cls(model):
    model_name = _get_full_model_name(model)
    assembler_cls = SUPPORTED_MODELS.get(model_name)

    if not assembler_cls:
        raise NotImplementedError(f"Model '{model_name}' is not supported")

    return assembler_cls
