
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

if __name__ == '__main__':

    #1. load data
    train = pd.read_csv("/datasets/titanic_train.csv")
    test = pd.read_csv("/datasets/titanic_test.csv")
    IDtest = test["PassengerId"]

    train_len = len(train)
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    dataset.drop(labels=["Name"], axis=1, inplace=True)
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})

    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
        else:
            Ticket.append("X")

    dataset["Ticket"] = Ticket

    print(dataset["Ticket"].head())
    dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
    dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
    dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(dataset)
    filled_values = imp_mean.transform(dataset)

    scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    scaled_data = scaler.fit_transform(filled_values)



    dataset = pd.DataFrame(scaled_data, columns=dataset.columns)




    print(dataset)
    train = dataset[:train_len]
    test = dataset[train_len:]
    test = test.drop(labels=["Survived"], axis=1, inplace=True)

    train["Survived"] = train["Survived"].astype(int)

    Y_train = train["Survived"]

    X_train = train.drop(labels=["Survived"], axis=1)

    #2. Cross Validation
    kfold = StratifiedKFold(n_splits=10)
    folds = kfold.split(X_train, Y_train)
    print(folds)

    #3. Modeling testing models
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                          learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                                                           "RandomForest", "ExtraTrees",
                                                                            "GradientBoosting",
                                                                             "KNeighbors",
                                                                            "LogisticRegression"]})
    print(cv_res)
    # Top 5
    number_jobs = 10
    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'probability': [True],
                      'C': [1, 10, 50, 100, 200, 300, 1000]}
    gb_param_grid = {'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }
    ada_param_grid = {"estimator__criterion": ["gini", "entropy"],
                      "estimator__splitter": ["best", "random"],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "n_estimators": [1, 2],
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}
    rf_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [ 'sqrt', 'log2']
    }
    KN_param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    LR_param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500]
    }
    estimators = []

    for classifier in classifiers:
        print(str(classifier))
        if str(classifier).startswith("SVC"):
            gsExtC = GridSearchCV(classifier, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        elif str(classifier).startswith("GradientBoosting"):
            gsExtC = GridSearchCV(classifier, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        elif str(classifier).startswith("ExtraTrees"):
            gsExtC = GridSearchCV(classifier, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        elif str(classifier).startswith("AdaBoost"):
            gsExtC = GridSearchCV(classifier, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        elif str(classifier).startswith("RandomForest"):
            gsExtC = GridSearchCV(classifier, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        elif str(classifier).startswith("LogisticRegression"):
            gsExtC = GridSearchCV(classifier, param_grid=LR_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        elif str(classifier).startswith("KNeighbors"):
            gsExtC = GridSearchCV(classifier, param_grid=KN_param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        else:
            gsExtC = GridSearchCV(classifier, param_grid=param_grid, cv=kfold, scoring="accuracy", n_jobs=number_jobs,
                                  verbose=1)
        gsExtC.fit(X_train, Y_train)
        index = str(classifier).find("(")
        name = str(classifier)[:index]
        estimators.append((name,gsExtC.best_estimator_ ))
        print(gsExtC.best_score_)
        print(gsExtC.best_estimator_)

    print(estimators)
    votingC = VotingClassifier(estimators=estimators, voting='soft',
                               n_jobs=4)

    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    # Sort DataFrame based on column 'A' in descending order and retrieve top 5 rows
    top_5 = cv_res.nlargest(5, 'CrossValMeans')

    print(top_5)
    # hyperpameter Tunning the best models