import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor

from Dictionary.CustomFunction.Taxi_DateTimeFeatures import CustomFeatureEngineer
from Dictionary.CustomFunction.Taxi_OneHot import CustomOneHotEncoder
from Dictionary.CustomFunction.Taxi_Outlier_Removal import Taxi_Outlier_Removal
from Dictionary.Ensemblers.CustomAverageEnsemble import GL__AverageRegressorLoader
from Dictionary.Ensemblers.CustomStackingEnsemble import GL__StackingRegressorLoader
from Dictionary.Ensemblers.StackingEnsemble import SetackingEnsemble
from Dictionary.Ensemblers.VotingEnsemble import VotingRegressorLoader
from Dictionary.Evaluation.ComputeAccuracy import AccuracyCalculator
from Dictionary.Evaluation.F1_score import F1ScoreCalculator
from Dictionary.Evaluation.KS import KS
from Dictionary.Evaluation.ComputeAUC import ComputeAUC
from Dictionary.Evaluation.MAECalculator import MAECalculator
from Dictionary.Evaluation.MPECalculator import MPECalculator
from Dictionary.Evaluation.MSECalculator import MSECalculator


from Dictionary.CustomFunction.GL_PolynomialFeatures import GL__PolynomialFeatures

from Dictionary.NLP.TF__MLP import TF__MLP
from Dictionary.NLP.TR__MLP import TR__MLP
from Dictionary.PCA.GPU__PCA import GPU__PCA
from Dictionary.PCA.PCA_TensorFlow import TF__PCA
from Dictionary.PCA.TR__PCA import TR__PCA
from Dictionary.Regressors.GL__DecisionTreeRegressor import GL__DecisionTreeRegressor
from Dictionary.Regressors.GL__KNeighborRegressor import GL__KNeighborsRegressor
from Dictionary.Regressors.GL__LGBM import GL__LGBMRegressor
from Dictionary.Regressors.GL__LassoGLM import GL__LassoGLM
from Dictionary.Regressors.TF_Ridge import TF__Ridge
from Dictionary.Regressors.TF__LinearRegressor import TF__LinearRegressor
from Dictionary.SVM.GL_LibSVMEstimator import GL__SVMLibEstimator
from Dictionary.SVM.SVM_PyTorch import TR__LinearSVC
from Dictionary.SVM.SVM_TensorFlow import TF__LnearSVC
from Dictionary.SVM.TF_SV import TF__LinearSVC
from Dictionary.SimpleImputer.GL_MeanImputer import GL__SimpleImputer
from Dictionary.SimpleImputer.GPU__SimpleImputer import GPU__SimpleImputer
from Dictionary.SimpleImputer.TF__SimpleImputer import TF__SimpleImputer
from Dictionary.SimpleImputer.TR__SimpleImputer import TR__SimpleImputer
from Dictionary.StandardScaler.GL_ScipyScaler import GL__StScaler
from Dictionary.StandardScaler.SS_GPU import GPU__StandardScaler
from Dictionary.StandardScaler.StandardScalerTensorFlow import TF__StandardScaler
from Dictionary.StandardScaler.TR_StandarScaler import TR__StandardScaler

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

def handle_infinities(X, cap_value):
    X = np.where(np.isinf(X), np.nan, X)  # Replace infinities with NaNs
    X = np.where(X > cap_value, cap_value, X)  # Cap values
    X = np.where(X < -cap_value, -cap_value, X)  # Cap values
    return X

# Custom transformer for capping and handling infinities
class InfinityHandler(BaseEstimator, TransformerMixin):
    def __init__(self, cap_value=float('1e10')):  # Use a sensible cap value for your dataset
        self.cap_value = cap_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return handle_infinities(X, self.cap_value)





collab_HIGGS_experiment = [
    ('imputer', [SimpleImputer(strategy='mean'), GL__SimpleImputer()]),
    ('feature_generation', [PolynomialFeatures(degree=2), GL__PolynomialFeatures()]),
    ('scaler', [StandardScaler(), GL__StScaler()]),
    ('classifier', [
       GL__SVMLibEstimator(C=5, tol=10), SVC(C=5, tol=10, kernel='linear')]),
    ('score', [
        F1ScoreCalculator('nan'), AccuracyCalculator('nan')])
]


collab_HIGGS_experiment_2 = [
    ('imputer', [SimpleImputer(strategy='mean'), GL__SimpleImputer()]),
    ('scaler', [StandardScaler(), GL__StScaler()]),
    ('classifier', [
       GL__SVMLibEstimator(C=5, tol=10), SVC(C=5, tol=10, kernel='linear')]),
    ('score', [
        F1ScoreCalculator('nan'), AccuracyCalculator('nan')])
]

categorical_features = ['store_and_fwd_flag', 'vendor_id']

from sklearn.ensemble import GradientBoostingRegressor

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'is_training_metric': True
}
collab_TAXI_all_operators= [
    ('SI', [SimpleImputer(strategy='mean'), GL__SimpleImputer(strategy='mean')]),
    ('OR', [Taxi_Outlier_Removal()]),
    ('OH', [CustomOneHotEncoder(categorical_features)]),
    ('SS', [StandardScaler(), GL__StScaler()]),
    ('FE', [CustomFeatureEngineer()]),
    ('LR', [LinearRegression(), TF__LinearRegressor()]),
    ('KNR', [
        KNeighborsRegressor(n_neighbors=25, weights='distance'), GL__KNeighborsRegressor(n_neighbors=25, weights='distance')]),
    ('RI', [Ridge(alpha=75.0), TF__Ridge(alpha=75.0)]),
    ('LGBM', [
        LGBMRegressor(**lgb_params),GL__LGBMRegressor(**lgb_params)]),
    ('LA',[Lasso(alpha=0.75), GL__LassoGLM(alpha=0.75)]),
    ('MSE', [
        MSECalculator('nan')]),
    ('MAE', [
        MAECalculator('nan')]),
    ('MPE', [
        MPECalculator('nan')])
]

collab_TAXI_all_operators_advance = [
    ('SI', [SimpleImputer(strategy='mean'), GL__SimpleImputer(strategy='mean')]),
    ('OR', [Taxi_Outlier_Removal()]),
    ('OH', [CustomOneHotEncoder(categorical_features)]),
    ('SS', [StandardScaler(), GL__StScaler()]),
    ('FE', [CustomFeatureEngineer()]),
    ('VR(2)', [
        GL__AverageRegressorLoader(models_dir="taxi_models_2", n_models=2)]),
    ('VR(3)', [
        GL__AverageRegressorLoader(models_dir="taxi_models_2", n_models=3)]),
    ('VR(4)', [
        GL__AverageRegressorLoader(models_dir="taxi_models_2", n_models=4)]),
    ('VR(5)', [
        GL__AverageRegressorLoader(models_dir="taxi_models_2", n_models=5)]),
    ('SE(2)', [
        GL__StackingRegressorLoader(models_dir="taxi_models_2", n_models=2)]),
    ('SE(3)', [
        GL__StackingRegressorLoader(models_dir="taxi_models_2", n_models=3)]),
    ('SE(4)', [
        GL__StackingRegressorLoader(models_dir="taxi_models_2", n_models=4)]),
    ('SE(5)', [
        GL__StackingRegressorLoader(models_dir="taxi_models_2", n_models=5)]),
    ('MSE', [
        MSECalculator('nan')]),
    ('MAE', [
        MAECalculator('nan')]),
    ('MPE', [
        MPECalculator('nan')])
]

collab_HIGGS_all_operators = [
    ('SI', [SimpleImputer(strategy='mean'), GL__SimpleImputer(strategy='mean')]),
    ('PF', [PolynomialFeatures(degree=2), GL__PolynomialFeatures(degree=2)]),
    ('SS', [StandardScaler(), GL__StScaler()]),
    ('SVM(0.05)', [
       TF__LinearSVC(C=0.05), LinearSVC(C=0.05)]),
    ('SVM(0.5)', [
        TF__LinearSVC(C=0.5), LinearSVC(C=0.5)]),
    ('SVM(1)', [
        TF__LinearSVC(C=1), LinearSVC(C=1)]),
    ('F1', [
        F1ScoreCalculator('nan')]),
    ('AC', [
        AccuracyCalculator('nan')]),
    ('CA', [
        ComputeAUC('nan')]),
    ('KS', [
        KS('nan')])

]



operators_dictionary = [
    ('SI', [SimpleImputer(strategy='mean'), GL__SimpleImputer(strategy='mean')]),
    ('PF', [PolynomialFeatures(degree=2), GL__PolynomialFeatures(degree=2)]),
    ('SS', [StandardScaler(), GL__StScaler()]),
    ('SVM(0.05)', [
       TF__LinearSVC(C=0.05), LinearSVC(C=0.05)]),
    ('SVM(0.5)', [
        TF__LinearSVC(C=0.5), LinearSVC(C=0.5)]),
    ('F1', [
        F1ScoreCalculator('nan')]),
    ('AC', [
        AccuracyCalculator('nan')]),
    ('CA', [
        ComputeAUC('nan')])
]




collab_HIGGS_single_platform = [
    ('SI', [SimpleImputer(strategy='mean')]),
    ('PF', [PolynomialFeatures(degree=2)]),
    ('SS', [StandardScaler()]),
    ('SVM(5)', [
        SVC(C=5, tol=10, kernel='linear')]),
    ('SVM(0.5)', [
         SVC(C=0.5, tol=10, kernel='linear')]),
    ('SVM(0.05)', [
        SVC(C=0.05, tol=10, kernel='linear')]),
    ('F1', [
        F1ScoreCalculator('nan')]),
    ('AC', [
        AccuracyCalculator('nan')])
]


collab_TAXI_all_operators_advance_old = [
    ('SI', [SimpleImputer(strategy='mean'), GL__SimpleImputer(strategy='mean')]),
    ('OR', [Taxi_Outlier_Removal()]),
    ('OH', [CustomOneHotEncoder(categorical_features)]),
    ('SS', [StandardScaler(), GL__StScaler()]),
    ('FE', [CustomFeatureEngineer()]),
    ('VR(2)', [
        VotingRegressorLoader(models_dir="taxi_models", n_models=2)]),
    ('VR(3)', [
        VotingRegressorLoader(models_dir="taxi_models", n_models=3)]),
    ('VR(4)', [
        VotingRegressorLoader(models_dir="taxi_models", n_models=4)]),
    ('VR(5)', [
        VotingRegressorLoader(models_dir="taxi_models", n_models=5)]),
    ('SE(2)', [
        SetackingEnsemble(models_dir="taxi_models", n_models=2)]),
    ('SE(3)', [
        SetackingEnsemble(models_dir="taxi_models", n_models=3)]),
    ('SE(4)', [
        SetackingEnsemble(models_dir="taxi_models", n_models=4)]),
    ('SE(5)', [
        SetackingEnsemble(models_dir="taxi_models", n_models=5)]),
    ('MSE', [
        MSECalculator('nan')]),
    ('MAE', [
        MAECalculator('nan')]),
    ('MPE', [
        MPECalculator('nan')])
]