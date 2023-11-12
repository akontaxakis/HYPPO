import random

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler
from sklearn.svm import SVC


def random_pipeline():
    # Define preprocessing steps and classifiers
    preprocessors = [
        ('standard_scaler', StandardScaler()),
        ('minmax_scaler', MinMaxScaler()),
        ('robust_scaler', RobustScaler()),
        ('pca', PCA(n_components=random.choice([5, 10, 15]))),
        ('poly_features', PolynomialFeatures(degree=random.choice([2, 3]))),
    ]
    classifiers = [
        ('logistic_regression', LogisticRegression()),
        ('sgd', SGDClassifier()),
        ('random_forest', RandomForestClassifier()),
        ('svc', SVC()),
        ('knn', KNeighborsClassifier()),
        ('naive_bayes', GaussianNB()),
    ]

    # Randomly select preprocessing steps and a classifier
    num_preprocessors = random.randint(0, 4)
    pipeline_steps = random.sample(preprocessors, num_preprocessors)
    pipeline_steps.append(random.choice(classifiers))

    return Pipeline(pipeline_steps)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    sum = 0;
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Generate 100 random pipelines
num_pipelines = 100
random_pipelines = [random_pipeline() for _ in range(num_pipelines)]

# Train and evaluate the pipelines
for i, pipeline in enumerate(random_pipelines):
    print(pipeline.steps)
    sum = sum + len(pipeline.steps)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Pipeline {i + 1} accuracy: {accuracy:.4f}")
print(sum)
