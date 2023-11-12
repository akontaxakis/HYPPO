import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target

    # Choose a class as the normal class (0: malignant, 1: benign)
    normal_class = 1
    y = np.where(y == normal_class, 1, -1)  # 1 for normal, -1 for anomalous

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Filter the training data for normal class samples
    X_train_normal = X_train[y_train == 1]

    # Train the Isolation Forest model
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_train_normal)
    y_pred = clf.predict(X_test)

    # Evaluate the model using classification metrics
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


