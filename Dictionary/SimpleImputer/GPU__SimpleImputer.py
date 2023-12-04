import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn.base import BaseEstimator, TransformerMixin

mod = SourceModule("""
__global__ void impute(float *data, float *statistics, int rows, int cols, float missing_value)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int row = idx / cols;
    int col = idx % cols;

    if (row < rows)
    {
        if (data[idx] == missing_value)
        {
            data[idx] = statistics[col];
        }
    }
}
""", options=["-Wno-deprecated-gpu-targets", "-Xcompiler", "-Wall"])

impute = mod.get_function("impute")

class GPU__SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=float('nan'), strategy='mean'):
        self.missing_values = missing_values
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        #print(type(X))
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0).astype(np.float32)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return self

    def transform(self, X):
        #print(type(X))
        rows, cols = X.shape
        X_gpu = gpuarray.to_gpu(np.array(X, dtype=np.float32))
        statistics_gpu = gpuarray.to_gpu(self.statistics_)

        impute(X_gpu, statistics_gpu, np.int32(rows), np.int32(cols), np.float32(self.missing_values),
               block=(1024, 1, 1), grid=((rows * cols + 1023) // 1024, 1))

        return X_gpu.get()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)