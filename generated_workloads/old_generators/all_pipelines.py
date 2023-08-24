
import itertools

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from skit_learn_components.GPU__PCA import GPU__PCA
from skit_learn_components.GPU__SimpleImputer import GPU__SimpleImputer
from skit_learn_components.PCA_TensorFlow import TF__PCA
from skit_learn_components.SVM_PyTorch import TR_SVM
from skit_learn_components.SVM_TensorFlow import TF_SVM
from skit_learn_components.StandardScalerTensorFlow import TF__StandardScaler
from skit_learn_components.TR__PCA import TR__PCA
from skit_learn_components.TR__SimpleImputer import TR__SimpleImputer
from skit_learn_components.TR_StandarScaler import TR__StandardScaler
from itertools import product


if __name__ == '__main__':

    test_sk_tf_trch_steps = [
        ('imputer', [GPU__SimpleImputer(strategy='mean'), TR__SimpleImputer(strategy='mean'), SimpleImputer(strategy='mean')]),
        ('scaler', [TR__StandardScaler(), TF__StandardScaler(), StandardScaler()]),
        ('dimensionality_reduction', [PCA(n_components=10), GPU__PCA(n_components=10), TF__PCA(n_components=10), TR__PCA(n_components=10)]),
        ('classifier', [
            TF_SVM(),
            TR_SVM(),
            LinearSVC(loss="hinge", max_iter = 100)
        ])
    ];

    steps_lists = [step[1] for step in test_sk_tf_trch_steps]

    # Generate all combinations of pipeline steps
    pipelines = []
    for steps_combination in product(*steps_lists):
        # Create a pipeline for this combination
        pipeline_steps = [(test_sk_tf_trch_steps[i][0], step) for i, step in enumerate(steps_combination)]
        pipeline = Pipeline(pipeline_steps)
        pipelines.append(pipeline)

    print(f"Generated {len(pipelines)} pipelines.")




