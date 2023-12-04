
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from Dictionary.NLP.TF__MLP import TF__MLP
from Dictionary.NLP.TR__MLP import TR__MLP
from Dictionary.PCA.GPU__PCA import GPU__PCA
from Dictionary.PCA.PCA_TensorFlow import TF__PCA
from Dictionary.PCA.TR__PCA import TR__PCA
from Dictionary.SVM.SVM_PyTorch import TR__LinearSVC
from Dictionary.SVM.SVM_TensorFlow import TF__LinearSVC
from Dictionary.SimpleImputer.GPU__SimpleImputer import GPU__SimpleImputer
from Dictionary.SimpleImputer.TF__SimpleImputer import TF__SimpleImputer
from Dictionary.SimpleImputer.TR__SimpleImputer import TR__SimpleImputer
from Dictionary.StandardScaler.SS_GPU import GPU__StandardScaler
from Dictionary.StandardScaler.StandardScalerTensorFlow import TF__StandardScaler
from Dictionary.StandardScaler.TR_StandarScaler import TR__StandardScaler

GPU_steps_CN = [
    ('imputer', [GPU__SimpleImputer(strategy='mean')]),
    ('scaler', [GPU__StandardScaler()]),
    ('dimensionality_reduction', [GPU__PCA(n_components=10)]),
    ('classifier', [
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)

    ])
]

TF_steps_CN = [
    ('imputer', [TF__SimpleImputer(strategy ='mean')]),
    ('scaler', [TF__StandardScaler()]),
    ('dimensionality_reduction', [TF__PCA(n_components=10)]),
    ('classifier', [
        TF__MLP()
    ])
]

TR_steps_CN = [
    ('imputer', [TR__SimpleImputer(strategy='mean')]),
    ('scaler', [TR__StandardScaler()]),
    ('dimensionality_reduction', [TR__PCA(n_components=10)]),
    ('classifier', [
        TR__MLP()
    ])
]

SK_steps_CN = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=10)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)
    ])
]




GPU_steps_CL = [
    ('imputer', [GPU__SimpleImputer(strategy='mean')]),
    ('scaler', [GPU__StandardScaler()]),
    ('dimensionality_reduction', [GPU__PCA(n_components=10)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)
        #MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)
    ])
]

TF_steps_CL = [
    ('imputer', [TF__SimpleImputer(strategy ='mean')]),
    ('scaler', [TF__StandardScaler()]),
    ('dimensionality_reduction', [TF__PCA(n_components=8)]),
    ('classifier', [
        TF__LinearSVC()
    ])
]

TR_steps_CL = [
    ('imputer', [TR__SimpleImputer(strategy='mean')]),
    ('scaler', [TR__StandardScaler()]),
    ('dimensionality_reduction', [TR__PCA(n_components=8)]),
    ('classifier', [
        TR__LinearSVC()
    ])
]

SK_steps_CL = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=10)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)
    ])
]


test_sk_tf_trch_steps = [
    ('imputer', [GPU__SimpleImputer(strategy='mean'), TR__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean')]),
    ('scaler', [TR__StandardScaler(), TF__StandardScaler(), StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=10), GPU__PCA(n_components=10), TF__PCA(n_components=10), TR__PCA(n_components=10)]),
    ('classifier', [
        TF__LinearSVC(),
        TR__LinearSVC(),
        LinearSVC(loss="hinge", max_iter = 100)
    ])
]

SK_steps_CN_1 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=10)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)
    ])
]


SK_steps_CN_2 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=10)]),
    ('classifier', [
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)])
]