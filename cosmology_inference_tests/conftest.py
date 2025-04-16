import pytest
import numpy as np

@pytest.fixture
def mock_radii():
    return np.linspace(0.1, 3.0, 50)

@pytest.fixture
def mock_density_profile(mock_radii):
    delta = -0.4 * np.exp(-mock_radii**2)
    sigma = 0.05 * np.ones_like(delta)
    return mock_radii, delta, sigma

