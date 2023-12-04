import pickle

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from libs.parser import create_equivalent_graph, plot_artifact_graph

if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")

    steps = [
        ('scaler', [StandardScaler()]),
        ('imputer', [SimpleImputer(strategy='mean')]),
        ('polynomial_features', [PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)]),
        ('feature_selection', [
            SelectKBest(f_classif, k=10),
            SelectPercentile(f_classif, percentile=50),
            SelectFromModel(Lasso(alpha=0.01)),
            SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        ('dimensionality_reduction', [PCA(n_components=10), PCA_GPU(n_components=10)]),
        ('classifier', [
            LogisticRegression(max_iter=10000, solver='liblinear'),
            KNeighborsClassifier(),
            SVC(),
            DecisionTreeClassifier(),
            RandomForestClassifier()
        ])
    ]

    uid = '1a6de191'
    ag = create_equivalent_graph(steps, uid, artifact_graph, metrics_dir='../metrics')
    plot_artifact_graph(ag, "eq_"+uid)