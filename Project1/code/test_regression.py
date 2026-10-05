"""Unit tests for regression.py.

Run with `pytest test_regression.py`, or with `python test_regression.py`
if pytest is not installed.
"""

import numpy as np
from sklearn.linear_model import Ridge

from regression import runge, make_data, polynomial_features, MSE, R2, ols, ridge, fit_predict


def _scaled_design(degree=6, n=100, seed=2026):
    x, y = make_data(n=n, sigma=0.1, seed=seed)
    X = polynomial_features(x, degree)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y - y.mean()


def test_runge_known_values():
    assert np.isclose(runge(0.0), 1.0)
    assert np.isclose(runge(0.2), 0.5)          # 1/(1 + 25*0.04)
    assert np.isclose(runge(1.0), runge(-1.0))  # even function


def test_make_data_is_reproducible():
    x1, y1 = make_data(n=50, seed=1)
    x2, y2 = make_data(n=50, seed=1)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_polynomial_features_columns():
    x = np.array([2.0, 3.0])
    X = polynomial_features(x, 3)
    assert np.allclose(X, [[2, 4, 8], [3, 9, 27]])


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert MSE(y, y) == 0.0
    assert np.isclose(R2(y, y), 1.0)
    assert np.isclose(R2(y, np.full(3, y.mean())), 0.0)


def test_ols_recovers_exact_polynomial():
    """Noise-free data from a degree-3 polynomial must give back its coefficients."""
    x = np.linspace(-1, 1, 30)
    theta_true = np.array([0.5, -2.0, 1.5])
    X = polynomial_features(x, 3)
    assert np.allclose(ols(X, X @ theta_true), theta_true)


def test_ridge_at_zero_penalty_equals_ols():
    X, y = _scaled_design(degree=6)
    assert np.allclose(ridge(X, y, 0.0), ols(X, y), atol=1e-10)


def test_ridge_matches_sklearn():
    """Our cost has 1/n in front of the squared error, so alpha = n*lambda."""
    X, y = _scaled_design(degree=6)
    lam = 0.01
    sk = Ridge(alpha=len(y) * lam, fit_intercept=False).fit(X, y).coef_
    assert np.allclose(ridge(X, y, lam), sk, atol=1e-12)


def test_ridge_shrinks_coefficients():
    X, y = _scaled_design(degree=10)
    norms = [np.linalg.norm(ridge(X, y, lam)) for lam in (1e-6, 1e-3, 1e-1, 10.0)]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_fit_predict_degree_one_on_a_line():
    x = np.linspace(-1, 1, 20)
    y = 3.0 * x + 1.0
    y_train_pred, y_test_pred, _ = fit_predict(x, x, y, degree=1)
    assert np.allclose(y_train_pred, y) and np.allclose(y_test_pred, y)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("passed:", t.__name__)
