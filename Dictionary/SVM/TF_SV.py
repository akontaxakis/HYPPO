import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TF__LinearSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='linear', C=1.0, max_iter=100, learning_rate=0.01):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def _fit_linear(self, X, y):
        n_samples, n_features = X.shape

        # Initialize TensorFlow variables
        self.W = tf.Variable(tf.zeros([n_features, 1]), name="weights")
        self.b = tf.Variable(tf.zeros([1]), name="bias")

        # Model input and output
        X_input = tf.constant(X, dtype=tf.float32)
        y_input = tf.constant(y.reshape(-1, 1), dtype=tf.float32)

        # Optimization
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)

        # Training
        for _ in range(self.max_iter):
            with tf.GradientTape() as tape:
                # Loss function: Hinge Loss
                model_output = tf.add(tf.matmul(X_input, self.W), self.b)
                regularizer = tf.reduce_sum(tf.square(self.W))
                hinge_loss = tf.reduce_sum(tf.maximum(0., 1. - model_output * y_input))
                loss = tf.add(hinge_loss, 0.5 * self.C * regularizer)

            gradients = tape.gradient(loss, [self.W, self.b])
            optimizer.apply_gradients(zip(gradients, [self.W, self.b]))

    def fit(self, X, y):
        X, y = check_X_y(X, y)  # Check if the input is valid
        self.classes_ = list(set(y))
        y[y == 0] = -1  # Change 0 to -1 for SVM

        if self.kernel == 'linear':
            self._fit_linear(X, y)
        else:
            raise NotImplementedError("Non-linear kernels are not implemented")

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)  # Validate the input

        # Make predictions
        X_input = tf.constant(X, dtype=tf.float32)
        model_output = tf.add(tf.matmul(X_input, self.W), self.b)
        predictions = tf.sign(model_output)

        return tf.cast(predictions, tf.int32).numpy()
