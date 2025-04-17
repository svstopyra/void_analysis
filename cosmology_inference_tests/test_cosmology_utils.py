# test_cosmology_utils.py

import numpy as np
import os
import pytest
from void_analysis.cosmology_inference import (
    Ez2,
    Hz,
    f_lcdm,
    ap_parameter
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# ---------------------- UNIT TESTS: Ez2 ----------------------

def test_Ez2_positive():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    Ez2_vals = Ez2(z, Om)
    assert np.all(Ez2_vals > 0)

def test_Ez2_monotonic_increasing():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    Ez2_vals = Ez2(z, Om)
    assert np.all(np.diff(Ez2_vals) >= 0)

# ---------------------- UNIT TESTS: Hz ----------------------

def test_Hz_positive():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    Hz_vals = Hz(z, Om)
    assert np.all(Hz_vals > 0)

def test_Hz_scaling_with_h():
    z = 0.5
    Om = 0.31
    h1 = 0.7
    h2 = 0.8
    hz1 = Hz(z, Om, h=h1)
    hz2 = Hz(z, Om, h=h2)
    assert np.isclose(hz2/hz1, h2/h1, rtol=1e-6)

# ---------------------- UNIT TESTS: f_lcdm ----------------------

def test_f_lcdm_typical_values():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    f_vals = f_lcdm(z, Om)
    assert np.all((f_vals >= 0) & (f_vals <= 1))

# ---------------------- UNIT TESTS: ap_parameter ----------------------

def test_ap_parameter_basic():
    z = np.linspace(0, 2, 50)
    Om = 0.3
    Om_fid = 0.3
    ap = ap_parameter(z, Om, Om_fid)
    assert np.allclose(ap, np.ones_like(z), rtol=1e-5)

# ---------------------- REGRESSION TESTS ----------------------

def test_Ez2_regression():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    Ez2_vals = Ez2(z, Om)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "Ez2_ref.npy"))
    np.testing.assert_allclose(Ez2_vals, ref, rtol=1e-6)

def test_Hz_regression():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    Hz_vals = Hz(z, Om)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "Hz_ref.npy"))
    np.testing.assert_allclose(Hz_vals, ref, rtol=1e-6)

def test_f_lcdm_regression():
    z = np.linspace(0, 2, 100)
    Om = 0.31
    f_vals = f_lcdm(z, Om)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "f_lcdm_ref.npy"))
    np.testing.assert_allclose(f_vals, ref, rtol=1e-6)

def test_ap_parameter_regression():
    z = np.linspace(0, 2, 50)
    Om = 0.31
    Om_fid = 0.3
    ap = ap_parameter(z, Om, Om_fid)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "ap_parameter_ref.npy"))
    np.testing.assert_allclose(ap, ref, rtol=1e-6)

