from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # Load the iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Create RandomForest and GradientBoosting classifiers
    rf_clf = KNeighborsClassifier(n_neighbors=5)
    gb_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

    # Train the classifiers
    rf_clf.fit(X_train, y_train)
    gb_clf.fit(X_train, y_train)

    # Make predictions
    rf_predictions = rf_clf.predict(X_test)
    gb_predictions = gb_clf.predict(X_test)

    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    gb_accuracy = accuracy_score(y_test, gb_predictions)

    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.3f}")

    # Load the iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Define base models
    base_models = [
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]

    # Create the StackingClassifier using LogisticRegression as the final estimator
    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

    # Train the classifier
    stacking_clf.fit(X_train, y_train)

    # Make predictions
    stacking_predictions = stacking_clf.predict(X_test)

    # Calculate accuracy
    stacking_accuracy = accuracy_score(y_test, stacking_predictions)

    print(f"Stacking Classifier Accuracy: {stacking_accuracy:.3f}")
