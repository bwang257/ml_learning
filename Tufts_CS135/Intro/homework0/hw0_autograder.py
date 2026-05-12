'''
hw0_autograder.py developed using Claude Code
A simulated autograder for hw0_split.py and hw0_knn.py.

Run:
    $ python hw0_autograder.py

Each test prints PASS/FAIL with a short description. A summary and a
weighted "grade" is printed at the end. This is not the real Tufts
autograder; it just exercises the behaviors specified in the docstrings
plus a handful of edge cases a real grader would likely cover.
'''
import doctest
import io
import sys
import traceback
import numpy as np

import Tufts_CS135_23.Intro.homework0.hw0_split as hw0_split
import Tufts_CS135_23.Intro.homework0.hw0_knn as hw0_knn

from Tufts_CS135_23.Intro.homework0.hw0_split import split_into_train_and_test
from Tufts_CS135_23.Intro.homework0.hw0_knn import calc_k_nearest_neighbors


# ----- tiny test harness -------------------------------------------------

_RESULTS = []  # list of (suite, name, passed, weight, message)

def check(suite, name, weight=1.0):
    def deco(fn):
        try:
            fn()
            _RESULTS.append((suite, name, True, weight, ""))
        except AssertionError as e:
            _RESULTS.append((suite, name, False, weight, str(e) or "assertion failed"))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            _RESULTS.append((suite, name, False, weight, f"{type(e).__name__}: {e}\n{tb}"))
        return fn
    return deco


# ----- doctests ----------------------------------------------------------

def run_doctests(suite, module):
    buf = io.StringIO()
    runner = doctest.DocTestRunner(verbose=False)
    finder = doctest.DocTestFinder()
    tests = finder.find(module, extraglobs={"np": np})
    failures = 0
    attempted = 0
    for t in tests:
        r = runner.run(t, out=buf.write)
        failures += r.failed
        attempted += r.attempted
    passed = (failures == 0 and attempted > 0)
    msg = "" if passed else f"{failures}/{attempted} doctest examples failed:\n{buf.getvalue()}"
    _RESULTS.append((suite, f"doctests ({attempted} examples)", passed, 2.0, msg))


# ----- split_into_train_and_test -----------------------------------------

@check("split", "shapes match expected M and N for frac_test=2/6", weight=1.0)
def _():
    x = np.arange(12).reshape(6, 2)
    tr, te = split_into_train_and_test(x, frac_test=2/6,
                                       random_state=np.random.RandomState(0))
    assert tr.shape == (4, 2), f"train shape {tr.shape}"
    assert te.shape == (2, 2), f"test shape {te.shape}"

@check("split", "test size rounds UP (ceil) when L*frac_test is non-integer", weight=1.0)
def _():
    x = np.arange(20).reshape(10, 2)
    _, te = split_into_train_and_test(x, frac_test=0.31,
                                      random_state=np.random.RandomState(0))
    assert te.shape[0] == 4, f"expected N=4 (ceil(10*0.31)), got {te.shape[0]}"

@check("split", "input array is not mutated by call", weight=1.0)
def _():
    x = np.asarray([[0, 11], [0, 22], [0, 33], [-2, 44], [-2, 55], [-2, 66]])
    before = x.copy()
    split_into_train_and_test(x, frac_test=2/6,
                              random_state=np.random.RandomState(0))
    assert np.array_equal(x, before), "input array was modified"

@check("split", "outputs are independent allocations (not views of input)", weight=1.0)
def _():
    x = np.arange(20).reshape(10, 2).copy()
    tr, te = split_into_train_and_test(x, frac_test=0.3,
                                       random_state=np.random.RandomState(0))
    assert tr.base is not x and (tr.base is None or tr.base is not x), \
        "train appears to be a view of input"
    # Mutate outputs and confirm input unchanged.
    snap = x.copy()
    tr[0, 0] = 9999
    te[0, 0] = 9999
    assert np.array_equal(x, snap), "mutating outputs changed input -> outputs are views"

@check("split", "train+test partition exactly covers all input rows", weight=1.5)
def _():
    rng = np.random.RandomState(7)
    x = rng.randn(50, 4)
    tr, te = split_into_train_and_test(x, frac_test=0.4,
                                       random_state=np.random.RandomState(1))
    # every row of x appears exactly once in (tr ∪ te), and nothing extra.
    combined = np.vstack([tr, te])
    # sort rows lex for comparison
    a = combined[np.lexsort(combined.T[::-1])]
    b = x[np.lexsort(x.T[::-1])]
    assert a.shape == b.shape and np.allclose(a, b), \
        "union of train+test does not equal input set"

@check("split", "deterministic for a given RandomState seed", weight=1.0)
def _():
    x = np.arange(30).reshape(15, 2)
    tr1, te1 = split_into_train_and_test(x, 0.3, np.random.RandomState(42))
    tr2, te2 = split_into_train_and_test(x, 0.3, np.random.RandomState(42))
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2), \
        "same seed produced different splits"

@check("split", "integer random_state is accepted (and seeds reproducibly)", weight=1.0)
def _():
    x = np.arange(30).reshape(15, 2)
    tr1, te1 = split_into_train_and_test(x, 0.3, random_state=123)
    tr2, te2 = split_into_train_and_test(x, 0.3, random_state=123)
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2), \
        "integer seed should be reproducible"

@check("split", "matches docstring example exactly", weight=1.5)
def _():
    x = np.asarray([[0, 11], [0, 22], [0, 33], [-2, 44], [-2, 55], [-2, 66]])
    tr, te = split_into_train_and_test(x, frac_test=2/6,
                                       random_state=np.random.RandomState(0))
    expected_tr = np.asarray([[-2, 66], [0, 33], [0, 22], [-2, 44]])
    expected_te = np.asarray([[0, 11], [-2, 55]])
    assert np.array_equal(tr, expected_tr), f"train mismatch:\n{tr}\nexpected:\n{expected_tr}"
    assert np.array_equal(te, expected_te), f"test mismatch:\n{te}\nexpected:\n{expected_te}"


# ----- calc_k_nearest_neighbors ------------------------------------------

@check("knn", "output shape is (Q, K, F)", weight=1.0)
def _():
    data = np.random.RandomState(0).randn(20, 3)
    query = np.random.RandomState(1).randn(5, 3)
    out = calc_k_nearest_neighbors(data, query, K=4)
    assert out.shape == (5, 4, 3), f"got {out.shape}"

@check("knn", "K=1 returns the single nearest neighbor", weight=1.0)
def _():
    data = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    query = np.asarray([[0.9, 0.0], [0.0, -0.9]])
    out = calc_k_nearest_neighbors(data, query, K=1)
    assert np.allclose(out[0, 0], [1, 0]), f"q0 got {out[0,0]}"
    assert np.allclose(out[1, 0], [0, -1]), f"q1 got {out[1,0]}"

@check("knn", "K=3 returns neighbors sorted by ascending distance", weight=1.5)
def _():
    data = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    query = np.asarray([[0.9, 0.0]])
    out = calc_k_nearest_neighbors(data, query, K=3)
    # distances from [0.9, 0]: to (1,0)=0.1, (0,1)=sqrt(.81+1), (0,-1)=sqrt(.81+1), (-1,0)=1.9
    # The first neighbor must be (1,0); then (0,1) and (0,-1) tie -> (0,1) first by row order.
    assert np.allclose(out[0, 0], [1, 0]), f"first neighbor wrong: {out[0,0]}"
    expected_rest = {(0.0, 1.0), (0.0, -1.0)}
    got_rest = {tuple(out[0, 1]), tuple(out[0, 2])}
    assert got_rest == expected_rest, f"2nd/3rd neighbors wrong: {got_rest}"

@check("knn", "tie-breaking: equal distances -> earlier row in data_NF wins", weight=1.5)
def _():
    # Both rows are equidistant from query at origin.
    data = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    query = np.asarray([[0.0, 0.0]])
    out = calc_k_nearest_neighbors(data, query, K=2)
    # Expect first two rows in data order since all are equidistant.
    assert np.allclose(out[0, 0], [1, 0]) and np.allclose(out[0, 1], [-1, 0]), \
        f"expected ties broken by row order, got {out[0]}"

@check("knn", "K=N returns all data points sorted by distance", weight=1.0)
def _():
    data = np.asarray([[3.0], [0.0], [1.0], [2.0]])
    query = np.asarray([[0.0]])
    out = calc_k_nearest_neighbors(data, query, K=4)
    expected = np.asarray([[[0.0], [1.0], [2.0], [3.0]]])
    assert np.allclose(out, expected), f"got {out}"

@check("knn", "matches docstring K=1 example", weight=1.0)
def _():
    data = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]])
    query = np.asarray([[0.9, 0], [0, -0.9]])
    out = calc_k_nearest_neighbors(data, query, K=1)
    assert out.shape == (2, 1, 2)
    assert np.allclose(out[0], [[1.0, 0.0]])
    assert np.allclose(out[1], [[0.0, -1.0]])

@check("knn", "matches docstring K=3 example", weight=1.5)
def _():
    data = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]])
    query = np.asarray([[0.9, 0], [0, -0.9]])
    out = calc_k_nearest_neighbors(data, query, K=3)
    assert out.shape == (2, 3, 2)
    assert np.allclose(out[0], [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), f"q0 got\n{out[0]}"
    assert np.allclose(out[1], [[0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]), f"q1 got\n{out[1]}"

@check("knn", "multi-feature, multi-query sanity check", weight=1.0)
def _():
    rng = np.random.RandomState(0)
    data = rng.randn(30, 5)
    query = rng.randn(7, 5)
    K = 6
    out = calc_k_nearest_neighbors(data, query, K=K)
    assert out.shape == (7, K, 5)
    # Verify by brute force for a couple of queries.
    for qi in [0, 3, 6]:
        d = np.linalg.norm(data - query[qi], axis=1)
        order = np.argsort(d, kind="stable")[:K]
        assert np.allclose(out[qi], data[order]), \
            f"query {qi} neighbors don't match brute force"


# ----- run + report ------------------------------------------------------

def main():
    run_doctests("split-doctests", hw0_split)
    run_doctests("knn-doctests", hw0_knn)
    # The @check decorators have already executed all tests at import time
    # of this module's body; just print the report.

    suites = {}
    for suite, name, ok, w, msg in _RESULTS:
        suites.setdefault(suite, []).append((name, ok, w, msg))

    total_w = 0.0
    earned_w = 0.0
    print("=" * 72)
    print("hw0 simulated autograder")
    print("=" * 72)
    for suite, items in suites.items():
        print(f"\n[{suite}]")
        for name, ok, w, msg in items:
            tag = "PASS" if ok else "FAIL"
            print(f"  {tag}  ({w:.1f})  {name}")
            if not ok and msg:
                # Indent the failure detail.
                for line in msg.rstrip().splitlines():
                    print(f"        {line}")
            total_w += w
            if ok:
                earned_w += w

    pct = 100.0 * earned_w / total_w if total_w else 0.0
    print("\n" + "=" * 72)
    print(f"Score: {earned_w:.1f} / {total_w:.1f}   ({pct:.1f}%)")
    print("=" * 72)
    # Exit code: 0 if all passed, 1 otherwise (handy for CI).
    sys.exit(0 if earned_w == total_w else 1)


if __name__ == "__main__":
    main()
