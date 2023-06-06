# This is a sample Python script.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import time

    # Generate a random dataset
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])

    # Define the hyperparameter grid
    param_grid = {
        'model__n_estimators': [10, 50, 100],
        'model__max_depth': [None, 5, 10]
    }

    # Perform the grid search
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    # Print the best hyperparameters
    print('Best hyperparameters: ', grid_search.best_params_)
    print('Execution time: %.2f seconds' % (end_time - start_time))

    # Print all the models found during the grid search
    results = pd.DataFrame(grid_search.cv_results_)
    print('All models found during grid search:')
    for i in range(len(results)):
        print(f"\nModel {i + 1}")
        print('Mean validation score: ', results.loc[i, 'mean_test_score'])
        print('Hyperparameters: ', results.loc[i, 'params'])

    # Predict on the test set using the best model
    y_pred = grid_search.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print('Test accuracy: ', accuracy)
