from cosmology_inference import log_likelihood_profile

def test_likelihood_basic(mock_density_profile):
    r, delta, sigma = mock_density_profile
    theta = [1.0, 1.0, 1.0, -0.2, 0.0, 1.0]
    ll = log_likelihood_profile(theta, r, delta, sigma,
                                profile_model=lambda r, *args: r*0 + 0.0)
    assert isinstance(ll, float)

