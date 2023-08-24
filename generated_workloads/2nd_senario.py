import random
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold

from libs.artifact_graph_lib import plot_artifact_graph, store_EDGES_artifact_graph
from skit_learn_components.ensemble_steps import detect_outliers, feature_analysis, modeling


def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size


if __name__ == '__main__':
    import os

    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")

    # Load data
    ##### Load train and Test set
    uid = "s2"
    dataset = "titanic"
    G = nx.DiGraph()

    start_time = time.time()
    train = pd.read_csv("C:/Users/adoko/PycharmProjects/pythonProject1/inputs/titanic_train.csv")
    test = pd.read_csv("C:/Users/adoko/PycharmProjects/pythonProject1/inputs/titanic_test.csv")
    end_time = time.time()
    G.add_edge("source", "titanic_train", weight=end_time - start_time + 0.000001, execution_time=end_time - start_time,
               memory_usage=0)
    G.add_edge("source", "titanic_test", weight=end_time - start_time + 0.000001, execution_time=end_time - start_time,
               memory_usage=0)

    ##### Outline Detection
    G.add_edge("titanic_train", "outline_detection", weight=0.000001, execution_time=0.000001,
               memory_usage=0)
    G.add_edge("titanic_test", "outline_detection", weight=0.000001, execution_time=0.000001,
               memory_usage=0)

    start_time = time.time()
    Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
    # Drop outliers
    train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

    train_len = len(train)
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    dataset = dataset.fillna(np.nan)
    end_time = time.time()

    G.add_edge("outline_detection", "dataset", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001,
               memory_usage=0)

    ##feature engineering
    start_time = time.time()
    dataset = feature_analysis(dataset, train)
    end_time = time.time()
    G.add_edge("dataset", "feature_engineering", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001,
               memory_usage=0)

    ##CrossValidation
    # G.add_node("transformed_train")
    # G.add_node("transformed_test")
    G.add_node("cross_validation")
    start_time = time.time()
    # Cross validate model with Kfold stratified cross val
    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=["Survived"], axis=1, inplace=True)
    train["Survived"] = train["Survived"].astype(int)
    Y_train = train["Survived"]
    X_train = train.drop(labels=["Survived"], axis=1)
    kfold = StratifiedKFold(n_splits=3)
    folds = kfold.split(X_train, Y_train)
    end_time = time.time()

    G.add_edge("dataset", "cross_validation", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001,
               memory_usage=0)

    # Modeling step Test differents algorithms
    start_time = time.time()
    keys = modeling(X_train, Y_train, kfold)

    print(keys)
    end_time = time.time()
    G.add_edge("cross_validation", "classifiers", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001,
               memory_usage=0)

    G.add_edge("classifiers", "SVC", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "DecisionTreeClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "AdaBoostClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "RandomForestClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "ExtraTreesClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "GradientBoostingClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "MLPClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "KNeighborsClassifier", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "LogisticRegression", weight=0.000001, execution_time=0.000001)
    G.add_edge("classifiers", "LinearDiscriminantAnalysis", weight=0.000001, execution_time=0.000001)

    ##2##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_2", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_2", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(estimators=[('rfc', keys[0]), ('extc', keys[1])], voting='soft', n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_2", "Voting_classifier_2", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##3##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_3", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_3", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_3", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ext23c', keys[2])], voting='soft',
                               n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_3", "Voting_classifier_3", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)
    ##4##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_4", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_4", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_4", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_4", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3])], voting='soft',
        n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_4", "Voting_classifier_4", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##5##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_5", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_5", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_5", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_5", weight=0.000001, execution_time=0.000001)
    G.add_edge("ExtraTreesClassifier", "ensemble_5", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3]),
                    ('ext4235c', keys[4])], voting='soft', n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_5", "Voting_classifier_5", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##6##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_6", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_6", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_6", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_6", weight=0.000001, execution_time=0.000001)
    G.add_edge("GradientBoostingClassifier", "ensemble_6", weight=0.000001, execution_time=0.000001)
    G.add_edge("ExtraTreesClassifier", "ensemble_6", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3]),
                    ('ext4235c', keys[4]), ('ex23c', keys[5])], voting='soft', n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_6", "Voting_classifier_6", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##7##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_7", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_7", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_7", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_7", weight=0.000001, execution_time=0.000001)
    G.add_edge("GradientBoostingClassifier", "ensemble_7", weight=0.000001, execution_time=0.000001)
    G.add_edge("MLPClassifier", "ensemble_7", weight=0.000001, execution_time=0.000001)
    G.add_edge("ExtraTreesClassifier", "ensemble_7", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3]),
                    ('ext4235c', keys[4]), ('ex23c', keys[5]), ('ex2323c', keys[6])], voting='soft', n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_7", "Voting_classifier_7", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##8##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("GradientBoostingClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("MLPClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("ExtraTreesClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    G.add_edge("KNeighborsClassifier", "ensemble_8", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3]),
                    ('ext4235c', keys[4]), ('ex23c', keys[5]), ('ex2323c', keys[6]), ('e33c', keys[7])], voting='soft',
        n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_8", "Voting_classifier_8", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##9##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("GradientBoostingClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("MLPClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("ExtraTreesClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("KNeighborsClassifier", "ensemble_9", weight=0.000001, execution_time=0.000001)
    G.add_edge("LogisticRegression", "ensemble_9", weight=0.000001, execution_time=0.000001)
    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3]),
                    ('ext4235c', keys[4]), ('ex23c', keys[5]), ('ex2323c', keys[6]), ('e33c', keys[7]),
                    ('ek33c', keys[8])], voting='soft',
        n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_9", "Voting_classifier_9", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    ##10##
    start_time = time.time()
    G.add_edge("SVC", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("DecisionTreeClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("AdaBoostClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("RandomForestClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("GradientBoostingClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("MLPClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("ExtraTreesClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("KNeighborsClassifier", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("LogisticRegression", "ensemble_10", weight=0.000001, execution_time=0.000001)
    G.add_edge("LinearDiscriminantAnalysis", "ensemble_10", weight=0.000001, execution_time=0.000001)

    votingC = VotingClassifier(
        estimators=[('rfc', keys[0]), ('extc', keys[1]), ('ex23tc', keys[2]), ('ext45c', keys[3]),
                    ('ext4235c', keys[4]), ('ex23c', keys[5]), ('ex2323c', keys[6]), ('e33c', keys[7]),
                    ('ek33c', keys[8]), ('ek2333c', keys[9])], voting='soft',
        n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    votingC.score(X_train, Y_train)
    print(votingC.score(X_train, Y_train))
    end_time = time.time()
    G.add_edge("ensemble_10", "Voting_classifier_10", weight=end_time - start_time + 0.000001,
               execution_time=end_time - start_time + 0.000001)

    plot_artifact_graph(G, uid, "equivalent")
    store_EDGES_artifact_graph(G, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")
