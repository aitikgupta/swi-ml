from itertools import combinations_with_replacement

try:
    import numpy as np
except ImportError:
    # exception already handled in backend
    pass


def index_combinations(degree, num_features):
    combinations = [
        combinations_with_replacement(range(num_features), i)
        for i in range(0, degree + 1)
    ]
    flat_combinations = [item for sublist in combinations for item in sublist]
    return flat_combinations


def transform_polynomial(X, degree):
    n_samples, n_features = np.shape(X)
    combinations = index_combinations(degree, n_features)
    n_output_features = len(combinations)

    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=-1)

    return X_new


def normalize(X):
    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(
        X[:, 1:], axis=0
    )
    return X
