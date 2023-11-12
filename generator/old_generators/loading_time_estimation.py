import os
import pickle
import time

if __name__ == '__main__':
    artifact_dir='artifacts'
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    equal_pairs = []
    sum = 0
    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        step_start_time = time.time()
        with open(file1, 'rb') as f:
            data1 = pickle.load(f)
            end_time = time.time()

        loading_time = (end_time-step_start_time)
        if loading_time == 0:
            loading_time = 0.001
        print(loading_time)
        loading_speed = os.path.getsize(file1)/loading_time
        #print(str(os.path.getsize(file1))+","+str((end_time-step_start_time)))
        sum = sum + loading_speed
    print("average loading time bytes/second")
    print(sum/num_files)

    import numpy as np
    from sklearn.decomposition import PCA
    import time

    # Generate random data with 10000 samples and 100 features
    X = np.random.rand(100000, 100)

    # Create a PCA model
    pca = PCA(n_components=2)

    # Time the fit operation
    start_time_fit = time.time()
    fitted_pca = pca.fit(X)
    end_time_fit = time.time()

    # Time the transform operation
    start_time_transform = time.time()
    transformed_data = fitted_pca.transform(X)
    end_time_transform = time.time()

    # Get the sizes in bytes
    fitted_operator_size = fitted_pca.__sizeof__()
    transformed_data_size = transformed_data.__sizeof__()

    with open("10000000fitted_pca.pkl", "wb") as f:
        pickle.dump(fitted_pca, f)
    with open("10000000transformed_data.pkl", "wb") as f:
        pickle.dump(transformed_data, f)

    print("Size of the fitted operator: {} bytes".format(fitted_operator_size))
    print("Size of the transformed data: {} bytes".format(transformed_data_size))

    fit_execution_time = end_time_fit - start_time_fit
    transform_execution_time = end_time_transform - start_time_transform

    print("Execution time of fit: {:.6f} seconds".format(fit_execution_time))
    print("Execution time of transform: {:.6f} seconds".format(transform_execution_time))

    import numpy as np
    from sklearn.linear_model import LinearRegression
    import time

    # Generate random data with 10000 samples and 100 features
    X = np.random.rand(100000, 100)
    y = np.random.rand(100000)

    # Create a Linear Regression model
    model = LinearRegression()

    # Time the fit operation
    start_time_fit = time.time()
    fitted_model = model.fit(X, y)
    end_time_fit = time.time()

    # Time the predict operation
    start_time_predict = time.time()
    predictions = fitted_model.predict(X)
    end_time_predict = time.time()

    # Get the sizes in bytes
    fitted_operator_size = fitted_model.__sizeof__()
    predictions_size = predictions.__sizeof__()

    print("Size of the fitted model: {} bytes".format(fitted_operator_size))
    print("Size of the predictions: {} bytes".format(predictions_size))

    fit_execution_time = end_time_fit - start_time_fit
    predict_execution_time = end_time_predict - start_time_predict

    # Store the fitted model using pickle
    with open("10000000fitted_model.pkl", "wb") as f:
        pickle.dump(fitted_model, f)
    with open("10000000predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print("Execution time of fit: {:.6f} seconds".format(fit_execution_time))
    print("Execution time of predict: {:.6f} seconds".format(predict_execution_time))