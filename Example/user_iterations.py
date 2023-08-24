
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from skit_learn_components.ComputeAccuracy import AccuracyCalculator
from skit_learn_components.F1_score import F1ScoreCalculator
from skit_learn_components.NLP.TF__MLP import TF__MLP
from skit_learn_components.NLP.TR__MLP import TR__MLP
from skit_learn_components.PCA.GPU__PCA import GPU__PCA
from skit_learn_components.PCA.PCA_TensorFlow import TF__PCA
from skit_learn_components.PCA.TR__PCA import TR__PCA
from skit_learn_components.SVM.SVM_PyTorch import TR__LinearSVC
from skit_learn_components.SVM.SVM_TensorFlow import TF__LinearSVC
from skit_learn_components.SimpleImputer.GPU__SimpleImputer import GPU__SimpleImputer
from skit_learn_components.SimpleImputer.TF__SimpleImputer import TF__SimpleImputer
from skit_learn_components.SimpleImputer.TR__SimpleImputer import TR__SimpleImputer
from skit_learn_components.StandardScaler.SS_GPU import GPU__StandardScaler
from skit_learn_components.StandardScaler.StandardScalerTensorFlow import TF__StandardScaler
from skit_learn_components.StandardScaler.TR_StandarScaler import TR__StandardScaler

UR1_steps_0 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)])
]

UR1_steps_1 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=100)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)
    ])
]


UR1_steps_2 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=100)]),
    ('classifier', [
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)])
]


UR1_steps_3 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=100)]),
    ('classifier', [
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)]),
    ('score', [
        F1ScoreCalculator('nan')])
]


UR1_steps_4 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=100)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)]),
    ('score', [
        F1ScoreCalculator('nan')])
]



UR1_steps_5 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=100)]),
    ('classifier', [
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10)]),
    ('score', [
        AccuracyCalculator('nan')])
]


UR1_steps_6 = [
    ('imputer', [SimpleImputer(strategy='mean')]),
    ('scaler', [ StandardScaler()]),
    ('dimensionality_reduction', [PCA(n_components=100)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter = 100)]),
    ('score', [
        AccuracyCalculator('nan')])
]

UR2_steps_0 = [
    ('imputer', [TR__SimpleImputer(strategy='mean')]),
    ('scaler', [TR__StandardScaler()]),
    ('classifier', [
        TR__LinearSVC()
    ])
]
UR2_steps_1 = [
    ('imputer', [TR__SimpleImputer(strategy='mean')]),
    ('scaler', [TR__StandardScaler()]),
    ('dimensionality_reduction', [TR__PCA(n_components=100)]),
    ('classifier', [
        TR__LinearSVC()
    ])
]

UR2_steps_2 = [
    ('imputer', [TR__SimpleImputer(strategy='mean')]),
    ('scaler', [TR__StandardScaler()]),
    ('dimensionality_reduction', [TR__PCA(n_components=100)]),
    ('classifier', [
        TR__MLP()
    ])
]


all_steps = [
    ('imputer', [GPU__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean'),TR__SimpleImputer(),TF__SimpleImputer()]),
    ('scaler', [StandardScaler(), GPU__StandardScaler(),TR__StandardScaler(),TF__StandardScaler()]),
    ('dimensionality_reduction',
     [PCA(n_components=100), TR__PCA(n_components=100),GPU__PCA(n_components=100)]),
    ('classifier', [
        LinearSVC(loss="hinge", max_iter=100),
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=10),
        TR__LinearSVC(),
        TR__MLP()]),
    ('score', [
        F1ScoreCalculator('nan'), AccuracyCalculator('nan')])
]






