
# load GPR from the parent directory

import GPR,MSD_functions,utils,Fit_MSD
import numpy as np
import jax
import jax.numpy as jnp


def test_MSDfit():
    np.random.seed(42)
    ntracks = 50
    noise = .5
    Gamma = 0.5
    J = 1
    alpha = 0.3
    n = 2.0
    ndims = 3

    tmin,tmax = 0,6*60 # minutes
    dt = 0.5 # minutes 
    ntimes = int((tmax - tmin) / dt) + 1
    times = np.linspace(tmin,tmax,ntimes)

    regressor = GPR.GPR(times,MSD_functions.Saturating_MSD,ndims)

    theta = np.array([Gamma,J,alpha]*ndims + [noise]*ndims)
    true_dat,noisy_dat = regressor.sample_prior(theta, ntracks,nanfrac=0.1)

    # run fit
    true_vals = np.array([Gamma,J,alpha] + [noise]*ndims)
    res = Fit_MSD.fit_msd(
        noisy_dat,
        true_vals*np.random.uniform(0.5,2,size=true_vals.shape),
        dt,MSD_functions.Saturating_MSD,230,bootstrap_method="track")

    assert np.allclose(res[0]["final_params"], true_vals, rtol=0.2)