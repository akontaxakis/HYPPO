import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, QuantileTransformer, \
    FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from skit_learn_components.GPU_PCA import GPU_PCA
from skit_learn_components.GPU_SS_PCA import GPU_StandardScaler__PCA
from skit_learn_components.GPU_SimpleImputer import GPU_SimpleImputer
from skit_learn_components.SS_GPU import GPU_StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE

steps = [
    ('scaler', [StandardScaler(), GPU_StandardScaler()]),
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('polynomial_features', [PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]),
    ('feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(Lasso(alpha=0.01)),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('dimensionality_reduction',
     [GPU_StandardScaler__PCA(n_components=10), GPU_PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        LogisticRegression(max_iter=10000, solver='liblinear'),
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]


ad_eq_steps = [
    ('tfidf', [TfidfVectorizer(stop_words='english', max_df=0.5, sublinear_tf=True)]),
    ('1.scaler', [StandardScaler(), GPU_StandardScaler()]),

    ('4.dimensionality_reduction',
     [ GPU_PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        LogisticRegression(max_iter=10000, solver='liblinear'),
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]

simple_eq_steps_bigger = [
    ('1.scaler', [StandardScaler(), GPU_StandardScaler()]),
    ('2.polynomial_features', [PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]),
    ('3.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(Lasso(alpha=0.01)),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('4.dimensionality_reduction',
     [ GPU_PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]
simple_eq_steps_bigger_2 = [
    ('1.scaler', [StandardScaler(), GPU_StandardScaler()]),
    ('2.polynomial_features', [PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]),
    ('3.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(Lasso(alpha=0.01)),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('4.dimensionality_reduction',
     [ GPU_PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]
simple_eq_steps = [
    ('1.scaler', [StandardScaler(), GPU_StandardScaler()]),
    ('2.dimensionality_reduction',
     [ GPU_PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        LogisticRegression(max_iter=10000, solver='liblinear'),
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]

simple_eq_steps_1 = [
    ('1. scaler_dimensionality_reduction',[GPU_StandardScaler__PCA(n_components=10)]),
    ('classifier', [
        LogisticRegression(max_iter=10000, solver='liblinear'),
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]

person_1_steps = [
    ('1.scaler', [StandardScaler()]),
    ('2.imputer', [SimpleImputer(strategy='mean')]),
    ('3.dimensionality_reduction', [PCA(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        DecisionTreeClassifier()
    ])
]
person_2_steps = [
    ('1.scaler', [GPU_StandardScaler(), MinMaxScaler(), RobustScaler()]),
    ('2.imputer', [ GPU_SimpleImputer(strategy='mean'), MissingIndicator(features='all')]),
    ('3.dimensionality_reduction', [GPU_PCA(n_components=10), TruncatedSVD(n_components=10)]),
    ('classifier', [
        RandomForestClassifier(),
        SVC()
    ])
]
person_3_steps = [
    ('1. scaler_dimensionality_reduction', [GPU_StandardScaler__PCA(n_components=10)]),
    ('classifier', [
        SVC()
    ])
]

AG_3_steps = [
    ('1.scaler', [StandardScaler(), GPU_StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer(n_quantiles=100, output_distribution='normal')]),
    ('2.imputer', [GPU_SimpleImputer(strategy='mean'),SimpleImputer(strategy='mean'),KNNImputer(n_neighbors=5), MissingIndicator(features='all')]),
    #('3.polynomial_features', [PolynomialFeatures(degree=3)]),
    ('3.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('4.dimensionality_reduction',
     [ PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        RandomForestClassifier(),
        DecisionTreeClassifier()
    ])
]
AG_3_steps = [
    ('1.scaler', [StandardScaler(), GPU_StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer(n_quantiles=100, output_distribution='normal')]),
    ('2.imputer', [GPU_SimpleImputer(strategy='mean'),SimpleImputer(strategy='mean'),KNNImputer(n_neighbors=5), MissingIndicator(features='all')]),
    #('3.polynomial_features', [PolynomialFeatures(degree=3)]),
    ('3.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('4.dimensionality_reduction',
     [ PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        RandomForestClassifier(),
        DecisionTreeClassifier()
    ])
]


steps_with_sampling = [
    ('1.scaler', [StandardScaler(), GPU_StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer(n_quantiles=100, output_distribution='normal')]),
    ('2.imputer', [GPU_SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean'), KNNImputer(n_neighbors=5),
                   MissingIndicator(features='all')]),
     #('3.polynomial_features', [PolynomialFeatures(degree=3)]),
    ('4.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('5.dimensionality_reduction',
     [ PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        RandomForestClassifier(),
        DecisionTreeClassifier()
    ])
]