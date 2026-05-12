'''
Autograder for Tufts CS135 Homework 1 (Regression)

Built using Claude Code

Tests the three implementation files:
    1. LeastSquaresLinearRegression.py
    2. LeastSquaresQuadraticRegression.py
    3. performance_metrics.py

Run with:
    python autograder.py
'''

import sys
import traceback
import numpy as np

try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

RESULTS = []


def run_test(name, points, fn):
    try:
        fn()
        RESULTS.append((name, points, points, "OK"))
        print(f"  [PASS] ({points}/{points})  {name}")
    except AssertionError as e:
        RESULTS.append((name, 0, points, f"FAIL: {e}"))
        print(f"  [FAIL] (0/{points})  {name}")
        print(f"         {e}")
    except Exception as e:
        RESULTS.append((name, 0, points, f"ERROR: {e}"))
        print(f"  [ERR ] (0/{points})  {name}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Tests for LeastSquaresLinearRegressor
# -----------------------------------------------------------------------------

def test_linear_docstring_example():
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    prng = np.random.RandomState(0)
    N = 100
    true_w_F = np.asarray([1.1, -2.2, 3.3])
    true_b = 0.0
    x_NF = prng.randn(N, 3)
    y_N = true_b + np.matmul(x_NF, true_w_F) + 0.03 * prng.randn(N)

    lr = LeastSquaresLinearRegressor()
    lr.fit(x_NF, y_N)

    expected_w = np.array([1.099, -2.202, 3.301])
    expected_b = -0.005
    assert np.allclose(lr.w_F, expected_w, atol=1e-3), \
        f"weights mismatch: got {lr.w_F}, expected {expected_w}"
    assert np.isclose(lr.b, expected_b, atol=1e-3), \
        f"bias mismatch: got {lr.b}, expected {expected_b}"


def test_linear_predict_shape():
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    prng = np.random.RandomState(1)
    x_NF = prng.randn(50, 4)
    y_N = prng.randn(50)
    lr = LeastSquaresLinearRegressor()
    lr.fit(x_NF, y_N)
    yhat = lr.predict(x_NF)
    assert yhat.shape == (50,), f"predict output should be (50,), got {yhat.shape}"


def test_linear_perfect_fit():
    """With no noise, weights and bias should be recovered exactly."""
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    prng = np.random.RandomState(42)
    true_w = np.array([2.5, -1.7, 0.8, 4.0])
    true_b = 3.14
    x_NF = prng.randn(200, 4)
    y_N = true_b + x_NF @ true_w
    lr = LeastSquaresLinearRegressor()
    lr.fit(x_NF, y_N)
    assert np.allclose(lr.w_F, true_w, atol=1e-8), \
        f"weights: got {lr.w_F}, expected {true_w}"
    assert np.isclose(lr.b, true_b, atol=1e-8), \
        f"bias: got {lr.b}, expected {true_b}"


def test_linear_single_feature():
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    prng = np.random.RandomState(7)
    x_NF = prng.randn(80, 1)
    y_N = 5.0 + 2.0 * x_NF[:, 0]
    lr = LeastSquaresLinearRegressor()
    lr.fit(x_NF, y_N)
    assert np.isclose(lr.b, 5.0, atol=1e-6), f"bias: got {lr.b}"
    assert np.isclose(lr.w_F[0], 2.0, atol=1e-6), f"weight: got {lr.w_F[0]}"


def test_linear_matches_sklearn():
    if not HAS_SKLEARN:
        print("    (skipping sklearn comparison - not installed)")
        return
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    prng = np.random.RandomState(123)
    x_NF = prng.randn(150, 5)
    y_N = prng.randn(150)
    lr = LeastSquaresLinearRegressor()
    lr.fit(x_NF, y_N)
    sk = SklearnLR().fit(x_NF, y_N)
    assert np.allclose(lr.w_F, sk.coef_, atol=1e-6), \
        f"weights diverge from sklearn: {lr.w_F} vs {sk.coef_}"
    assert np.isclose(lr.b, sk.intercept_, atol=1e-6), \
        f"bias diverges from sklearn: {lr.b} vs {sk.intercept_}"


def test_linear_attributes_exist():
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    lr = LeastSquaresLinearRegressor()
    x_NF = np.random.randn(20, 2)
    y_N = np.random.randn(20)
    lr.fit(x_NF, y_N)
    assert hasattr(lr, "w_F"), "missing attribute w_F"
    assert hasattr(lr, "b"), "missing attribute b"
    assert lr.w_F.shape == (2,), f"w_F should be (2,), got {lr.w_F.shape}"
    assert np.isscalar(lr.b) or np.ndim(lr.b) == 0, \
        f"b should be a scalar, got shape {np.shape(lr.b)}"


# -----------------------------------------------------------------------------
# Tests for LeastSquaresQuadraticRegressor
# -----------------------------------------------------------------------------

def test_quadratic_docstring_example():
    from LeastSquaresQuadraticRegression import LeastSquaresQuadraticRegressor
    prng = np.random.RandomState(0)
    N = 100
    true_w_F = np.asarray([1.1, -2.2])
    true_b = 0.0
    x_NF = prng.randn(N, 2)
    x_NF[:, 1] = x_NF[:, 0] ** 2
    y_N = true_b + np.matmul(x_NF, true_w_F) + 0.03 * prng.randn(N)
    x_N = x_NF[:, 0]

    qr = LeastSquaresQuadraticRegressor()
    qr.fit(x_N, y_N)

    expected_w = np.array([1.101, -2.194])
    expected_b = -0.008
    assert np.allclose(qr.w_F, expected_w, atol=1e-3), \
        f"weights: got {qr.w_F}, expected {expected_w}"
    assert np.isclose(qr.b, expected_b, atol=1e-3), \
        f"bias: got {qr.b}, expected {expected_b}"


def test_quadratic_perfect_fit():
    """y = b + w1*x + w2*x^2, no noise."""
    from LeastSquaresQuadraticRegression import LeastSquaresQuadraticRegressor
    prng = np.random.RandomState(11)
    true_w = np.array([0.5, 3.0])
    true_b = -1.25
    x_N = prng.randn(150)
    y_N = true_b + true_w[0] * x_N + true_w[1] * x_N ** 2
    qr = LeastSquaresQuadraticRegressor()
    qr.fit(x_N, y_N)
    assert np.allclose(qr.w_F, true_w, atol=1e-8), \
        f"weights: got {qr.w_F}, expected {true_w}"
    assert np.isclose(qr.b, true_b, atol=1e-8), \
        f"bias: got {qr.b}, expected {true_b}"


def test_quadratic_predict_shape():
    from LeastSquaresQuadraticRegression import LeastSquaresQuadraticRegressor
    prng = np.random.RandomState(2)
    x_N = prng.randn(60)
    y_N = prng.randn(60)
    qr = LeastSquaresQuadraticRegressor()
    qr.fit(x_N, y_N)
    yhat = qr.predict(x_N)
    assert yhat.shape == (60,), f"predict shape: got {yhat.shape}, expected (60,)"


def test_quadratic_predict_values():
    """Check that predict reproduces y exactly when fit is perfect."""
    from LeastSquaresQuadraticRegression import LeastSquaresQuadraticRegressor
    prng = np.random.RandomState(99)
    x_N = prng.randn(100)
    y_N = 2.0 + 0.3 * x_N - 1.5 * x_N ** 2
    qr = LeastSquaresQuadraticRegressor()
    qr.fit(x_N, y_N)
    yhat = qr.predict(x_N)
    assert np.allclose(yhat, y_N, atol=1e-8), \
        f"predict diverged: max abs error {np.max(np.abs(yhat - y_N))}"


def test_quadratic_get_features():
    from LeastSquaresQuadraticRegression import LeastSquaresQuadraticRegressor
    qr = LeastSquaresQuadraticRegressor()
    x_N = np.array([1.0, 2.0, 3.0, -1.0])
    feats = qr.get_quadratic_features(x_N)
    assert feats.shape == (4, 2), f"shape: {feats.shape}"
    assert np.allclose(feats[:, 0], x_N), "first column should be x"
    assert np.allclose(feats[:, 1], x_N ** 2), "second column should be x^2"


# -----------------------------------------------------------------------------
# Tests for calc_root_mean_squared_error
# -----------------------------------------------------------------------------

def test_rmse_scalar_input():
    from performance_metrics import calc_root_mean_squared_error
    rmse = calc_root_mean_squared_error(0.0, 4.123)
    assert np.isclose(float(rmse), 4.123, atol=1e-6), \
        f"got {rmse}, expected 4.123"


def test_rmse_docstring_example():
    from performance_metrics import calc_root_mean_squared_error
    y_N = np.asarray([-2, 0, 2], dtype=np.float64)
    yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
    rmse = calc_root_mean_squared_error(y_N, yhat_N)
    assert np.isclose(float(rmse), 1.154701, atol=1e-5), \
        f"got {rmse}, expected ~1.154701"


def test_rmse_zero_when_equal():
    from performance_metrics import calc_root_mean_squared_error
    y_N = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rmse = calc_root_mean_squared_error(y_N, y_N)
    assert np.isclose(float(rmse), 0.0, atol=1e-12), \
        f"rmse on identical arrays should be 0, got {rmse}"


def test_rmse_known_value():
    from performance_metrics import calc_root_mean_squared_error
    y_N = np.array([1.0, 2.0, 3.0, 4.0])
    yhat_N = np.array([2.0, 3.0, 4.0, 5.0])
    # diff = 1 each, mse = 1, rmse = 1
    rmse = calc_root_mean_squared_error(y_N, yhat_N)
    assert np.isclose(float(rmse), 1.0, atol=1e-10), \
        f"rmse: got {rmse}, expected 1.0"


def test_rmse_larger_errors():
    from performance_metrics import calc_root_mean_squared_error
    y_N = np.array([0.0, 0.0, 0.0, 0.0])
    yhat_N = np.array([3.0, 4.0, 0.0, 0.0])
    # mse = (9+16)/4 = 6.25, rmse = 2.5
    rmse = calc_root_mean_squared_error(y_N, yhat_N)
    assert np.isclose(float(rmse), 2.5, atol=1e-10), \
        f"rmse: got {rmse}, expected 2.5"


# -----------------------------------------------------------------------------
# Integration test: linear regressor + RMSE
# -----------------------------------------------------------------------------

def test_integration_linear_rmse_low():
    from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
    from performance_metrics import calc_root_mean_squared_error
    prng = np.random.RandomState(0)
    true_w = np.array([1.1, -2.2, 3.3])
    x_NF = prng.randn(100, 3)
    y_N = x_NF @ true_w + 0.03 * prng.randn(100)
    lr = LeastSquaresLinearRegressor()
    lr.fit(x_NF, y_N)
    yhat = lr.predict(x_NF)
    rmse = float(calc_root_mean_squared_error(y_N, yhat))
    assert rmse < 0.1, f"rmse should be small, got {rmse}"


def test_integration_quadratic_rmse_low():
    from LeastSquaresQuadraticRegression import LeastSquaresQuadraticRegressor
    from performance_metrics import calc_root_mean_squared_error
    prng = np.random.RandomState(0)
    x_N = prng.randn(100)
    y_N = 1.1 * x_N - 2.2 * x_N ** 2 + 0.03 * prng.randn(100)
    qr = LeastSquaresQuadraticRegressor()
    qr.fit(x_N, y_N)
    yhat = qr.predict(x_N)
    rmse = float(calc_root_mean_squared_error(y_N, yhat))
    assert rmse < 0.1, f"rmse should be small, got {rmse}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CS135 Homework 1 Autograder")
    print("=" * 70)

    print("\n[1] LeastSquaresLinearRegressor")
    run_test("linear: docstring example",          5, test_linear_docstring_example)
    run_test("linear: attributes w_F and b exist", 3, test_linear_attributes_exist)
    run_test("linear: predict output shape",       2, test_linear_predict_shape)
    run_test("linear: perfect fit recovers params",5, test_linear_perfect_fit)
    run_test("linear: single feature",             3, test_linear_single_feature)
    run_test("linear: matches sklearn",            5, test_linear_matches_sklearn)

    print("\n[2] LeastSquaresQuadraticRegressor")
    run_test("quad:   docstring example",          5, test_quadratic_docstring_example)
    run_test("quad:   get_quadratic_features",     3, test_quadratic_get_features)
    run_test("quad:   perfect fit recovers params",5, test_quadratic_perfect_fit)
    run_test("quad:   predict output shape",       2, test_quadratic_predict_shape)
    run_test("quad:   predict reproduces y",       3, test_quadratic_predict_values)

    print("\n[3] performance_metrics.calc_root_mean_squared_error")
    run_test("rmse:   scalar input (docstring)",   3, test_rmse_scalar_input)
    run_test("rmse:   array input (docstring)",    3, test_rmse_docstring_example)
    run_test("rmse:   zero when y == yhat",        2, test_rmse_zero_when_equal)
    run_test("rmse:   known value (constant err)", 3, test_rmse_known_value)
    run_test("rmse:   larger asymmetric errors",   3, test_rmse_larger_errors)

    print("\n[4] Integration")
    run_test("integration: linear + rmse",         3, test_integration_linear_rmse_low)
    run_test("integration: quadratic + rmse",      3, test_integration_quadratic_rmse_low)

    earned = sum(r[1] for r in RESULTS)
    total = sum(r[2] for r in RESULTS)
    n_pass = sum(1 for r in RESULTS if r[1] == r[2])

    print()
    print("=" * 70)
    print(f"Result: {n_pass}/{len(RESULTS)} tests passed   "
          f"Score: {earned}/{total}")
    print("=" * 70)

    failed = [r for r in RESULTS if r[1] != r[2]]
    if failed:
        print("\nFailed tests:")
        for name, _, _, status in failed:
            print(f"  - {name}: {status}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
