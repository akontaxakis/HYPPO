from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# Load the breast cancer dataset
from skit_learn_components.GPU_PCA import GPU_PCA
if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")

    data = load_breast_cancer().data

    # Create a scikit-learn pipeline that uses CPU PCA
    cpu_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2))
    ])

    # Create a scikit-learn pipeline that uses GPU PCA
    gpu_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", GPU_PCA(n_components=2))
    ])

    # Measure the time it takes to fit and transform the data using CPU PCA
    start_time = time.time()
    cpu_pipeline.fit_transform(data)
    cpu_time = time.time() - start_time

    # Measure the time it takes to fit and transform the data using GPU PCA
    start_time = time.time()
    gpu_pipeline.fit_transform(data)
    gpu_time = time.time() - start_time

    print("CPU time: {:.3f}s".format(cpu_time))
    print("GPU time: {:.3f}s".format(gpu_time))