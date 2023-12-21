
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


from Dictionary.Evaluation.F1_score import F1ScoreCalculator
from Dictionary.NLP.TF__MLP import TF__MLP
from Dictionary.NLP.TR__MLP import TR__MLP
from Dictionary.PCA.GPU_SS_PCA import GPU_StandardScaler__PCA
from Dictionary.PCA.GPU__PCA import GPU__PCA
from Dictionary.PCA.PCA_TensorFlow import TF__PCA
from Dictionary.PCA.TR__PCA import TR__PCA
from Dictionary.SVM.SVM_PyTorch import TR__LinearSVC
from Dictionary.SVM.TF_SV import TF__LinearSVC
from Dictionary.SimpleImputer.GPU__SimpleImputer import GPU__SimpleImputer
from Dictionary.StandardScaler.SS_GPU import GPU__StandardScaler
from Dictionary.StandardScaler.StandardScalerTensorFlow import TF__StandardScaler
from Dictionary.StandardScaler.TR_StandarScaler import TR__StandardScaler

steps = [
    ('scaler', [StandardScaler(), GPU__StandardScaler()]),
    ('impute', [SimpleImputer(strategy='mean')]),
    ('polynomial_features', [PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]),
    ('feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(Lasso(alpha=0.01)),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('dimensionality_reduction',
     [GPU_StandardScaler__PCA(n_components=10), GPU__PCA(n_components=10), PCA(n_components=10),
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
    ('1.scaler', [StandardScaler(), GPU__StandardScaler()]),

    ('4.dimensionality_reduction',
     [GPU__PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        LogisticRegression(max_iter=10000, solver='liblinear'),
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]

simple_eq_steps_bigger_2 = [
    ('1.scaler', [StandardScaler(), GPU__StandardScaler()]),
    ('2.polynomial_features', [PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]),
    ('3.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(Lasso(alpha=0.01)),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('4.dimensionality_reduction',
     [GPU__PCA(n_components=10), PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ])
]

simple_eq_steps = [
    ('1.scaler', [StandardScaler(), GPU__StandardScaler()]),
    ('2.dimensionality_reduction',
     [GPU__PCA(n_components=10), PCA(n_components=10),
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
    ('1. scaler_dimensionality_reduction', [GPU_StandardScaler__PCA(n_components=10)]),
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
    ('1.scaler', [GPU__StandardScaler(), MinMaxScaler(), RobustScaler()]),
    ('2.imputer', [GPU__SimpleImputer(strategy='mean'), MissingIndicator(features='all')]),
    ('3.dimensionality_reduction', [GPU__PCA(n_components=10), TruncatedSVD(n_components=10)]),
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
    ('1.scaler', [StandardScaler(), GPU__StandardScaler(), MinMaxScaler(), RobustScaler(),
                  QuantileTransformer(n_quantiles=100, output_distribution='normal')]),
    ('2.imputer', [GPU__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean'), KNNImputer(n_neighbors=5),
                   MissingIndicator(features='all')]),
    # ('3.polynomial_features', [PolynomialFeatures(degree=3)]),
    ('3.feature_selection', [
        SelectKBest(f_classif, k=10),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    ('4.dimensionality_reduction',
     [PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        RandomForestClassifier(),
        DecisionTreeClassifier()
    ])
]

all_steps = [
    ('scaler', [StandardScaler(), GPU__StandardScaler(), MinMaxScaler(), RobustScaler(),
                QuantileTransformer(n_quantiles=100, output_distribution='normal')]),
    ('imputer', [GPU__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean'), KNNImputer(n_neighbors=5),
                 TR__StandardScaler, TF__StandardScaler]),
    ('feature_selection', [
        SelectKBest(f_classif, k=12),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(RandomForestRegressor(n_estimators=20, random_state=42))
    ]),
    ('dimensionality_reduction',
     [PCA(n_components=10),
      TruncatedSVD(n_components=10)]), TR__PCA(n_components=10), GPU__PCA(n_components=10),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        TR__LinearSVC,
        TF__LinearSVC,
        TR__MLP(),
        TF__MLP(),
        LinearSVC(loss="hinge", max_iter=100),
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)]),
    ('score', [
        F1ScoreCalculator('nan')])
]

steps_with_sampling = [
    ('imputer', [GPU__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean'), KNNImputer(n_neighbors=5),
                 MissingIndicator(features='all')]),
    ('scaler', [StandardScaler(), GPU__StandardScaler(), MinMaxScaler(), RobustScaler(),
                QuantileTransformer(n_quantiles=100, output_distribution='normal')]),

    ('feature_selection', [
        SelectKBest(f_classif, k=12),
        SelectPercentile(f_classif, percentile=50),
        SelectFromModel(RandomForestRegressor(n_estimators=20, random_state=42))
    ]),
    ('dimensionality_reduction',
     [PCA(n_components=10),
      TruncatedSVD(n_components=10)]),
    ('classifier', [
        KNeighborsClassifier(),
        SVC(),
        RandomForestClassifier(),
        DecisionTreeClassifier()
    ])
]


sk_tf_trch_steps = [
    ('scaler', [TF__StandardScaler(), StandardScaler()]),
    ('imputer', [GPU__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean')]),
    ('dimensionality_reduction', [PCA(n_components=10), GPU__PCA(n_components=10), TF__PCA(n_components=10)]),
    ('classifier', [
        TF__LinearSVC(),
        TR__LinearSVC(),
        SVC(kernel="linear")

    ])
]
