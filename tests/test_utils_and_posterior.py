import numpy as np
import jax.numpy as jnp
from TwoLocusGPR import utils
from TwoLocusGPR import Posterior_analysis as pa


def test_get_mat_for_cholesky_masks_nans():
    data = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])
    covmat = jnp.stack([jnp.eye(2), jnp.eye(2)])
    noise = jnp.array([0.1, 0.2])

    nan_entries, masked_data, mats = utils.get_mat_for_cholesky(data, covmat, noise)

    assert nan_entries.shape == data.shape
    assert nan_entries[0, 1]
    assert masked_data[0, 1] == 0.0

    # large noise injected for NaN locations, finite otherwise
    assert mats[1, 0, 0] > 1e9
    assert np.isclose(mats[1, 1, 1], 1.0 + 2 * noise[1] ** 2)


def test_gaussian_radius_percentiles_monotonic():
    mu = jnp.zeros((2, 3, 3))
    sigma2 = 0.5 * jnp.ones_like(mu)

    pct = pa.gaussian_radius_percentiles(mu, sigma2, num_samples=400, seed=0)

    assert pct.shape == (2, 3, 3)
    assert np.all(pct[..., 0] <= pct[..., 1])
    assert np.all(pct[..., 1] <= pct[..., 2])


def test_prob_in_sphere_batch_bounds():
    mu = jnp.zeros((2, 3, 4))
    sigma = jnp.ones_like(mu)

    probs = pa.prob_in_sphere_batch(mu, sigma, R=1.5, num_samples=300)

    assert probs.shape == (2, 4)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
