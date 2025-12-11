
import GPR,MSD_functions
import numpy as np

def test_predictions():
        
    ntracks = 10
    noise = [48,47,120] # µm
    Gamma = 2.4*1e3*np.sqrt(60)/3 # µm^2/sqrt(min)
    J = (357)**2/3 # µm^2
    ndims = 3
    nanfrac=0.1
    tmin,tmax = 0,6.0*60 # minutes
    dt = 1/3 # minutes (20 sec)
    ntimes = int((tmax - tmin) / dt) + 1
    fine_factor = 10
    times = np.linspace(tmin,tmax,ntimes*fine_factor)

    regressor = GPR.GPR(times,MSD_functions.Rouse_MSD,ndims)

    theta = np.array([Gamma,J]*ndims + noise)
    true_dat,noisy_dat = regressor.sample_prior(theta, ntracks,nanfrac=nanfrac,seed=12)
    downsampled_times = times[::fine_factor]
    downsampled_noisy_dat = noisy_dat[:,::fine_factor]

    # perform posterior prediction
    pred_regressor = GPR.GPR(downsampled_times,MSD_functions.Rouse_MSD,ndims)
    theta = np.array([Gamma,J] + noise)
    p_mean,p_var = pred_regressor.Predict(theta,times,downsampled_noisy_dat)
    z_scores = ((p_mean - true_dat) / np.sqrt(p_var)).reshape(-1, ndims) # shape (ntracks*ntimes, ndims)
    
    # check that z-scores are standard normal
    assert np.allclose(np.nanmean(z_scores, axis=0), 0, atol=0.1)
    assert np.allclose(np.nanstd(z_scores, axis=0), 1, atol=0.1)

