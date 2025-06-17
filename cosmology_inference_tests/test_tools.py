# test_tools.py

import numpy as np
import pytest
import os
from void_analysis.tools import *

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def gaussian_likelihood():
    np.random.seed(0)
    N = 5
    mu = np.random.rand(5)
    A = np.random.randn(5, 5)
    C = A @ A.T
    return mu, C

# ---------------------- UNIT TESTS: -------------------------------------------

def test_gaussian_log_likelihood_function_negative(gaussian_likelihood):
    mu, C = gaussian_likelihood
    np.random.seed(1)
    data = np.random.rand(5)
    LL = gaussian_log_likelihood_function(data,mu,C)
    assert(LL < 0)

def test_get_finite_range_basic():
    ranges = [
        [0, np.inf], [-np.inf, np.inf], [-np.inf, 0],
        [-1, 0]
    ]
    target = ranges = [
        [0, 1], [-1, 1], [-1, 0],
        [-1, 0]
    ]
    for range_list, target_list in zip(ranges, target):
        computed = get_finite_range(range_list)
        assert(np.all(np.isfinite(computed)))
        for x,y in zip(target_list,computed):
            assert(x == y)

# ---------------------- REGRESSION TESTS --------------------------------------

def test_gaussian_log_likelihood_function(gaussian_likelihood):
    mu, C = gaussian_likelihood
    np.random.seed(1)
    data = np.random.rand(5)
    run_basic_regression_test(
        gaussian_log_likelihood_function,
        os.path.join(SNAPSHOT_DIR, "gaussian_log_likelihood_function_ref.npy"),
        data,mu,C
    )
