import os
import pickle
from pickle import dump

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def feature_analysis(dataset,train):
    #Fill Fare missing values with the median value
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    # Apply log to Fare to reduce skewness distribution
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    train[["Sex", "Survived"]].groupby('Sex').mean()
    #Fill Embarked nan values of dataset set with 'S' most frequent value
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    # convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
    for i in index_NaN_age:
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][(
                    (dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (
                        dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred):
            dataset['Age'].iloc[i] = age_pred
        else:
            dataset['Age'].iloc[i] = age_med
    # Get Title from Name
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    # Convert to categorical values Title
    dataset["Title"] = dataset["Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    dataset["Title"] = dataset["Title"].map(
        {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
    dataset["Title"] = dataset["Title"].astype(int)
    # Drop Name variable
    dataset.drop(labels=["Name"], axis=1, inplace=True)
    # Create a family size descriptor from SibSp and Parch
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
    # Create new feature of family size
    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
    # convert to indicator values Title and Embarked
    dataset = pd.get_dummies(dataset, columns=["Title"])
    dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
    # Replace the Cabin number by the type of cabin 'X' if not
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
    dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
    ## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  #Take prefix
        else:
            Ticket.append("X")
    dataset["Ticket"] = Ticket
    dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
    # Create categorical values for Pclass
    dataset["Pclass"] = dataset["Pclass"].astype("category")
    dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
    # Drop useless variables
    dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
    return dataset


def train_classifier(X_train, Y_train, kfold):
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state, probability=True))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                          learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    cv_results = {}
    for classifier in classifiers:
        cv_results[classifier] = cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold,
                                                 n_jobs=4).mean()
    print(cv_results)

    return list(cv_results.keys())

def modeling(X_train, Y_train, kfold):
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state, probability=True))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                          learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    model_dir = 'titanic_models_2'
    os.makedirs(model_dir, exist_ok=True)

    cv_results = {}
    for i, classifier in enumerate(classifiers):
        classifier.fit(X_train, Y_train)  # Fit the model
        cv_results[classifier] = cross_val_score(classifier, X_train, Y_train, scoring="accuracy", cv=kfold,
                                                 n_jobs=4).mean()

        # Save the model using pickle
        with open(os.path.join(model_dir, f'model_{i}.pickle'), 'wb') as f:
            pickle.dump(classifier, f)
    return list(cv_results.keys())
