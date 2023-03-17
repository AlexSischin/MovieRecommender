from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow import GradientTape

from data import read_ratings_as_sparse_matrix, export_collaborative_features


class CollaborativeRecommender(tf.Module):

    def __init__(self, name=None):
        super().__init__(name)

    def compile(self,
                feature_alloc,
                x_alloc,
                w_alloc,
                learning_rate,
                regularization,
                batch_size_limit):
        x_indices, w_indices = get_slices(batch_size_limit, x_alloc, w_alloc)
        self._X = tf.Variable(tf.random.normal([x_alloc, feature_alloc], dtype=tf.float64), name='X')
        self._W = tf.Variable(tf.random.normal([w_alloc, feature_alloc], dtype=tf.float64), name='W')
        self._b = tf.Variable(tf.random.normal([w_alloc], dtype=tf.float64), name='b')
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._x_slices = list(zip(x_indices[:-1], x_indices[1:]))
        self._w_slices = list(zip(w_indices[:-1], w_indices[1:]))

    @property
    def X(self):
        return self._X

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @tf.function
    def _compute_cost(self, X_i, W_j, b_j, Y_ij, R_ij, R_number):
        print('__Tracing__')
        Y_ij_hat = tf.matmul(X_i, W_j, transpose_b=True) + b_j
        loss = tf.reduce_sum((Y_ij - Y_ij_hat * R_ij) ** 2) / R_number
        penalty = tf.reduce_sum(X_i ** 2) + tf.reduce_sum(W_j ** 2)
        cost = loss + penalty * self._regularization
        return cost, loss

    def fit(self, Y: csr_matrix, epochs=10):
        R = (Y != 0).astype(float)
        for epoch in range(epochs):
            for x_a, x_b in self._x_slices:
                X_i = tf.Variable(self._X[x_a:x_b], name='X_i', dtype=tf.float64)
                for w_a, w_b in self._w_slices:
                    W_j = tf.Variable(self._W[w_a:w_b], name='W_j', dtype=tf.float64)
                    b_j = tf.Variable(self._b[w_a:w_b], name='b_j', dtype=tf.float64)
                    Y_ij = tf.constant(Y[x_a:x_b:, w_a:w_b].todense(), dtype=tf.float64)
                    R_ij_sparse = R[x_a:x_b:, w_a:w_b]
                    R_number = tf.constant(R_ij_sparse.data.size, dtype=tf.float64)
                    R_ij = tf.constant(R_ij_sparse.todense(), dtype=tf.float64)

                    if R_number > 0:
                        with GradientTape() as tape:
                            cost, loss = self._compute_cost(X_i, W_j, b_j, Y_ij, R_ij, R_number)

                        local_variables = [X_i, W_j, b_j]
                        gradients = tape.gradient(cost, local_variables)
                        optimizer = tf.keras.optimizers.Adam(self._learning_rate)
                        optimizer.apply_gradients(zip(gradients, local_variables))

                        self._W[w_a:w_b].assign(W_j)
                        self._b[w_a:w_b].assign(b_j)
                    else:
                        loss = tf.constant(0)

                    print(f'Epoch={epoch}, X_range=[{x_a},{x_b}), W_range=[{w_a},{w_b}), MSE loss={loss.numpy()}')
                self._X[x_a:x_b].assign(X_i)

    @tf.function
    def _compute_error_sum(self, X_i, W_j, b_j, Y_ij, R_ij):
        print('__Tracing__')
        Y_ij_hat = tf.matmul(X_i, W_j, transpose_b=True) + b_j
        return tf.reduce_sum((Y_ij - Y_ij_hat * R_ij) ** 2)

    def evaluate_mse(self, Y: csr_matrix):
        example_number = Y.data.size
        error_sum = 0
        R = (Y != 0).astype(int)
        for x_a, x_b in self._x_slices:
            X_i = self._X[x_a:x_b]
            for w_a, w_b in self._w_slices:
                Y_ij = tf.constant(Y[x_a:x_b:, w_a:w_b].todense(), dtype=tf.float64)
                W_j = self._W[w_a:w_b]
                b_j = self._b[w_a:w_b]
                R_ij_sparse = R[x_a:x_b:, w_a:w_b]
                R_number = tf.constant(R_ij_sparse.data.size, dtype=tf.float64)
                R_ij = tf.constant(R_ij_sparse.todense(), dtype=tf.float64)
                if R_number > 0:
                    error_sum += self._compute_error_sum(X_i, W_j, b_j, Y_ij, R_ij)
                print(f'X_range=[{x_a},{x_b}), W_range=[{w_a},{w_b}), Square error sum={error_sum}')
        return error_sum / example_number


def get_slices(A_lim: np.int64, a: np.int64, b: np.int64) -> tuple[np.ndarray, np.ndarray]:
    a_slice_size_real = np.sqrt(A_lim * a / b)
    b_slice_size_real = np.sqrt(A_lim * b / a)
    a_slice_number = np.int64(np.ceil(a / a_slice_size_real))
    b_slice_number = np.int64(np.ceil(b / b_slice_size_real))
    a_slices = np.linspace(0, a, a_slice_number + 1, dtype=np.int64)
    b_slices = np.linspace(0, b, b_slice_number + 1, dtype=np.int64)
    return a_slices, b_slices


def main():
    train_ratings, dev_ratings, test_ratings = read_ratings_as_sparse_matrix()
    movie_allocation_number = max([s.shape[0] for s in [train_ratings, dev_ratings, test_ratings]])
    user_allocation_number = max([s.shape[1] for s in [train_ratings, dev_ratings, test_ratings]])
    feature_number = 10
    learning_rate = 0.3
    regularization = 10
    batch_size_limit = 50 * 1024 ** 2
    epoch_count = 15

    model = CollaborativeRecommender()
    model.compile(feature_number, movie_allocation_number, user_allocation_number, learning_rate, regularization,
                  batch_size_limit)

    start = timer()
    model.fit(train_ratings, epochs=epoch_count)
    mse_train = model.evaluate_mse(train_ratings)
    mse_dev = model.evaluate_mse(dev_ratings)
    end = timer()

    print(f'Time elapsed for training and evaluating: {((end - start) / 60):.1f} minutes')
    print(f'MSE (train): {mse_train}')
    print(f'MSE (dev): {mse_dev}')

    export_collaborative_features(model.X, model.W, model.b)


if __name__ == '__main__':
    main()
