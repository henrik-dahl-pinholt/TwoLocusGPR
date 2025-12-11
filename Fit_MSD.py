"""Bootstrap and fit MSD parameters using GP-based likelihoods."""

from ctypes import util
import matplotlib.pyplot as plt
import numpy as np
import jax
from scipy.linalg import solve_triangular

from scipy.optimize import minimize

from jax import grad, numpy as jnp
from tqdm.auto import tqdm

import utils


def bootstrap_MSD(arr_dat, block_size, nsamples=10_000, kind="track"):
    """Bootstrap MSD covariance across trajectories or blocks.

    Parameters
    ----------
    arr_dat : array-like, shape (ntraj, T, ndims)
        Input trajectories (NaNs allowed for missing data).
    block_size : int
        Block length (timepoints) for MSD estimation and block bootstrap.
    nsamples : int, optional
        Number of bootstrap replicates.
    kind : {"track", "block"}
        Resample whole tracks or contiguous blocks within tracks.

    Returns
    -------
    covariance_matrix : jnp.ndarray, shape (block_size, block_size, ndims)
        Bootstrap covariance of MSD estimates per dimension.
    mean : jnp.ndarray, shape (block_size, ndims)
        Bootstrap mean MSD per dimension.
    """
    ntraj = arr_dat.shape[0]
    T = arr_dat.shape[1]
    nblocks = T // block_size + 1
    lags = jnp.arange(1,block_size)
    ndims = arr_dat.shape[2]

    if kind=="block":
        def get_maxind(arr):
            for i in range(len(arr) - 1, -1, -1):
                if not np.any(np.isnan(arr[i])):
                    return i - block_size + 1
            return arr


        max_inds = jnp.array([get_maxind(arr) for arr in arr_dat])
        # generate mask of nans to generate varying lengths in the final output
        output_mask = jnp.ones_like(arr_dat)
        for i, ind in enumerate(max_inds):
            output_mask = output_mask.at[i, ind:].set(jnp.nan)

        @jax.jit
        def sample_block_inds(key):
            traj = jax.random.randint(key, shape=(), minval=0, maxval=ntraj)
            max_ind = max_inds[traj]
            key, subkey = jax.random.split(key)
            block_start = jax.random.randint(key, shape=(), minval=0, maxval=max_ind)
            return traj, block_start

        @jax.jit
        def sample_blocks(key):
            traj_ind, start_inds = jax.vmap(sample_block_inds)(key)
            aranges = jnp.arange(block_size)[None, :] + start_inds[:, None]
            data_sample = arr_dat[traj_ind[:, None], aranges]
            return data_sample


        @jax.jit
        def sample_experiment(key):
            keys = jax.random.split(key, (nblocks, ntraj))
            blocks = jax.vmap(sample_blocks)(keys).transpose(
                1, 0, 2,3
            )  # (ntraj, nblocks, block_size, ndims)
            exp = blocks.reshape(ntraj, nblocks * block_size, ndims)[:, :T]
            output = exp * output_mask
            return output
    else:  # kind=="track"
        @jax.jit
        def sample_experiment(key):
            traj_inds = jax.random.randint(
                key, shape=(ntraj,), minval=0, maxval=ntraj
            )
            return arr_dat[traj_inds]

    key = jax.random.PRNGKey(np.random.randint(0, 1e6))

    @jax.jit
    def scan(carry, key):
        n, mean, cov = carry
 

        bootstrap_sample = sample_experiment(key) # (ntraj,T,ndims)

        # compute acf
        result = utils.msd(bootstrap_sample, lags)[:block_size] # (block_size,ndims)
        track_avg = 2*jnp.nanmean(bootstrap_sample**2,axis=(0,1)) # (ndims,)
        result = jnp.concatenate([ result,track_avg[None, :]], axis=0)  # (block_size+1,ndims)
        # update state
        n += 1
        newmean = mean + (result - mean) / n
        newcov = cov + (result - mean)[:, None] * (result - newmean)[None, :] 
        return (n, newmean, newcov), result

    init_carry = (
        0,
        jnp.zeros((block_size , ndims)),
        jnp.zeros((block_size , block_size , ndims)),
    )
     
    for i in tqdm(list(range(nsamples))):
        key, subkey = jax.random.split(key)
        init_carry, _ = scan(init_carry, subkey)
    n, mean, cov = init_carry
    covariance_matrix = cov / (n - 1)
    return covariance_matrix, mean



def _profile_one_param(
    loss, loss_grad, hvp, x_hat_full, fix_idx, grid_vals_natural, method="Powell"
):
    """Profile a single parameter over a grid while optimizing the rest.

    Parameters
    ----------
    loss : callable
        Objective in log space.
    loss_grad : callable
        Gradient of ``loss``.
    hvp : callable
        Hessian-vector product of ``loss``.
    x_hat_full : np.ndarray
        Best-fit parameters in log space.
    fix_idx : int
        Index of the parameter to fix during profiling.
    grid_vals_natural : array-like
        Grid of values (natural scale) to scan for the fixed parameter.
    method : str, optional
        SciPy optimizer for the reduced problem.

    Returns
    -------
    dict
        Contains natural-scale grid, profiled loss values, and profiled
        parameter vectors in natural scale.
    """
    grid_log = np.log(np.asarray(grid_vals_natural))
    prof_losses = []
    prof_params_full_nat = []

    # warm start: begin from x_hat_full; then reuse previous solution
    x_free_current = np.delete(np.asarray(x_hat_full), fix_idx)
    pbar = tqdm(total=len(grid_log))
    pbar.set_description(f"Profiling param {fix_idx}")

    def runfit(gv_log, x0):
        free_idx, assemble_full, f_red, g_red, hvp_red = _make_restricted(
            loss, loss_grad, hvp, x_hat_full, fix_idx, gv_log
        )

        # wrap hvp into scipy's hessp signature
        def hessp_wrap(x_free, v_free):
            return hvp_red(x_free, v_free)

        res = minimize(
            f_red,
            x_free_current,
            method=method,
            # jac=g_red,
            # hessp=hessp_wrap,  # , tol=1e-10
        )
        return res, free_idx

    # do right_side first
    right_val = []
    right_params_full_nat = []
    for gv_log in grid_log[len(grid_log) // 2 :]:
        res, free_idx = runfit(gv_log, x_free_current)
        x_free_current = res.x  # warm start next

        # collect profiled optimum in FULL parameter space (natural)
        x_full_prof_log = np.array(x_hat_full, copy=True)
        x_full_prof_log[free_idx] = res.x
        x_full_prof_log[fix_idx] = gv_log

        # prof_losses.append(float(res.fun))
        right_params_full_nat.append(np.exp(x_full_prof_log))
        right_val.append(float(res.fun))

        pbar.update(1)
    # then left side, reset warm start
    x_free_current = np.delete(np.asarray(x_hat_full), fix_idx)
    left_val = []
    left_params_full_nat = []
    for gv_log in grid_log[: len(grid_log) // 2][::-1]:
        res, free_idx = runfit(gv_log, x_free_current)
        x_free_current = res.x  # warm start next

        # collect profiled optimum in FULL parameter space (natural)
        x_full_prof_log = np.array(x_hat_full, copy=True)
        x_full_prof_log[free_idx] = res.x
        x_full_prof_log[fix_idx] = gv_log

        # prof_losses.append(float(res.fun))
        left_params_full_nat.append(np.exp(x_full_prof_log))
        left_val.append(float(res.fun))

        pbar.update(1)
    pbar.close()
    prof_losses = left_val[::-1] + right_val
    prof_params_full_nat = left_params_full_nat[::-1] + right_params_full_nat

    return {
        "grid_natural": np.asarray(grid_vals_natural),
        "profile_loss": np.asarray(prof_losses),  # this is your profiled NLL
        "profile_params_full": np.asarray(prof_params_full_nat),  # exp(log x)
    }


def _make_restricted(loss, loss_grad, hvp, x_full0, fix_idx, fix_val_log):
    """Pin one parameter and build reduced objective/grad/HVP.

    Parameters
    ----------
    loss : callable
        Objective in log space.
    loss_grad : callable
        Gradient of ``loss``.
    hvp : callable
        Hessian-vector product of ``loss``.
    x_full0 : np.ndarray
        Reference parameter vector for sizing.
    fix_idx : int
        Index to hold fixed.
    fix_val_log : float
        Fixed value (log scale) for ``fix_idx``.

    Returns
    -------
    tuple
        ``(free_idx, assemble_full, f_red, g_red, hvp_red)`` where each entry
        operates in the reduced parameter space.
    """
    P = x_full0.size
    free_idx = np.array([i for i in range(P) if i != fix_idx], dtype=int)

    def assemble_full(x_free):
        x_full = np.array(x_full0, copy=True)
        x_full[free_idx] = x_free
        x_full[fix_idx] = fix_val_log
        return x_full

    # Reduced objective: f(x_free) = loss( scatter(x_free, fix=const) )
    def f_red(x_free):
        return float(loss(assemble_full(x_free)))

    # Reduced gradient via chain rule: g_free = (∂f/∂x_full)[free_idx]
    def g_red(x_free):
        g_full = np.array(loss_grad(assemble_full(x_free)))
        return g_full[free_idx]

    # Reduced HVP: H_free v = (H_full [S v])[free_idx], where S injects into free block
    def hvp_red(x_free, v_free):
        # build a full v with zeros everywhere except free_idx
        v_full = np.zeros_like(x_full0)
        v_full[free_idx] = v_free
        Hv_full = np.array(hvp(assemble_full(x_free), v_full))
        return Hv_full[free_idx]

    return free_idx, assemble_full, f_red, g_red, hvp_red



def fit_msd(
    arr_dat,
    init_guess,
    dt,
    msd_function,
    max_lag,
    nsamples=10_000,
    n_attempts=50,
    init_scale=3.0,
    bootstrap_method="track",
    inds_to_profile=None,
    profile=True,
):
    """Fit MSD model parameters by bootstrapped GP likelihood.

    Parameters
    ----------
    arr_dat : array-like, shape (ntraj, T, ndims)
        Input trajectories.
    init_guess : array-like
        Initial natural-scale parameter guess (MSD params + per-dim noise).
    dt : float
        Time step between observations.
    msd_function : callable
        Function ``msd(t, params)`` returning MSD at lags ``t``.
    max_lag : int
        Maximum lag (in time steps) for MSD calculation and covariance.
    nsamples : int, optional
        Number of bootstrap replicates for covariance estimation.
    n_attempts : int, optional
        Number of multi-start optimizations.
    init_scale : float, optional
        Log-normal perturbation scale for random initializations.
    bootstrap_method : {"track", "block"}
        Whether to resample full tracks or blocks.
    inds_to_profile : iterable[int] or None
        Parameter indices (log space) to profile for CIs; None profiles all.
    profile : bool, optional
        Whether to compute profile likelihood CIs.

    Returns
    -------
    results : dict
        Summary for best fit (trajectories, params, LLH, covariances, profiles).
    fit_results : list[dict]
        All attempted fits with trajectories and final parameters.
    LLHs : list[float]
        Final log-likelihoods (negative loss) per attempt.
    """

    lags = jnp.arange(1,max_lag)
    covar_matrix, _ = bootstrap_MSD(arr_dat, max_lag, nsamples=nsamples, kind=bootstrap_method) # (max_lag+1,max_lag+1,ndims)

    corrs = utils.msd(arr_dat, lags) # (max_lag,ndims)
    
    symmetrized_covar = 0.5 * (covar_matrix + covar_matrix.transpose((1,0,2))) + 1e-6 * jnp.eye(
        len(covar_matrix)
    )[..., None]
    
    L = jnp.linalg.cholesky(symmetrized_covar.transpose((2,0,1)))  # (ndims,max_lag+1,max_lag+1)
    
    data_term = jnp.concatenate([corrs,jnp.array([2*jnp.nanmean(arr_dat**2,axis=(0,1))])]) # (max_lag+1,ndims)

    times = dt * lags
    ndims = arr_dat.shape[2]
    
    @jax.jit
    def MSDfunc(params):
        noises = jnp.exp(params[-ndims:])
        params = jnp.exp(params[:-ndims])
        msd = msd_function(times,params)
        distance = msd_function(1e9,params)
        msd_part = jnp.concatenate([ msd,jnp.array([distance])])
        return msd_part[:,None] + 4*noises[None,:]**2 # (max_lag+1,ndims)


    @jax.jit
    def loss(x):

        corrs_model = MSDfunc(x) # (max_lag+1,ndims)

        diff = data_term - corrs_model # (max_lag+1,ndims)
        z = jax.scipy.linalg.solve(L, diff.T, lower=True) # (ndims, max_lag+1)
        val = 0.5 * jnp.vecdot(z, z).sum()
        # if nan or inf return large value
        return jnp.where(jnp.isfinite(val), val, 1e10)
    
    def gen_guess(guess):
        return np.exp(np.log(guess) + np.random.normal(scale=init_scale))

    
    def sample_guess():

        return np.log(
            np.array(
                [gen_guess(val) for val in init_guess
                    
                ]
            )
        )

    loss_grad = jax.jit(jax.grad(loss))

    # make function to compute hessian vector product
    @jax.jit
    def hvp(x, v):
        return jax.grad(lambda x: jnp.vdot(loss_grad(x), v))(x)

    pbar = tqdm(total=n_attempts)
    fit_results = []
    LLHs = []
    nfailed = 0
    nsucess = 0
    while nsucess < n_attempts:
        # pbar2 = tqdm(leave=False)
        try:
            guess = sample_guess()
            # make pbar that will dissapear after done

            trajectory = []
            paramtrajs = []

            def callback(xk):
                # pbar2.update()
                llh = -loss(xk)
                # pbar2.set_description(f"LLH: {llh:.2f}")
                trajectory.append(llh)
                paramtrajs.append(np.exp(xk))
                if len(LLHs) > 0:
                    pbar.set_description(
                        f"Best llh: {np.max(LLHs):.2f}, nfailed: {nfailed}, current LLH: {llh:.2f}"
                    )
                else:
                    pbar.set_description(
                        f"Best llh: {'-inf'}, nfailed: {nfailed}, current LLH: {llh:.2f}"
                    )

            res = minimize(
                loss,
                guess,
                method="Nelder-Mead",
                callback=callback,
                options={"maxiter": 100},
            )
            res = minimize(
                loss,
                res.x,
                method="trust-krylov",
                hessp=hvp,
                # hess=hess,
                jac=loss_grad,
                callback=callback,
            )
         
            results = {}
            results["LLH_trajectory"] = trajectory
            results["param_trajectory"] = paramtrajs
            results["final_LLH"] = -loss(res.x)
            # results["evidence"] = full_evidence
            results["final_params"] = np.exp(res.x)
            results["result_obj"] = res
            # pbar2.close()

            fit_results.append(results)
            LLHs.append(-res.fun)
            # pbar.set_description(f"Best LLH: {np.max(LLHs):.2f}, nfailed: {nfailed}")
            pbar.update()
            nsucess += 1
        except Exception as e:
            # if exception says: ValueError: array must not contain infs or NaNs move forward, else raise
            if "array must not contain infs or NaNs" in str(e):
                # pbar2.set_description("NaN/Inf in optimization, retrying...")
                # pbar2.close()
                pbar.set_description(
                    f"Best llh: {np.max(LLHs) if len(LLHs)>0 else -np.inf:.2f}, nfailed: {nfailed+1}"
                )
                

                nfailed += 1
                continue
            else:
                raise e
    pbar.close()
    best_ind = np.argmax(LLHs)
    results = fit_results[best_ind]
    res = results["result_obj"]

    results["covar_matrix"] = covar_matrix
    results["data msd"] = data_term
    results["predicted msd"] = MSDfunc(res.x)
    results["msd_func"] = MSDfunc
    
    best_ind = np.argmax(LLHs)
    results = fit_results[best_ind]
    res = results["result_obj"]
    x_hat = res.x  # log-parameters at the best run

    if profile:
        # compute profile likelihoods of params
        CIs = []
        if inds_to_profile is None:
            inds_to_profile = np.arange(len(x_hat))
        # for i in range(len(x_hat)):
        for i in np.arange(len(res.x))[inds_to_profile]:
            if not jnp.isfinite(x_hat[i]):
                raise ValueError("Non-finite parameter in MLE, cannot profile")
            # create grid around MLE
            factors = np.geomspace(0.25, 4, 25)  # 25 points from 0.25x to 4x
            grid = np.exp(x_hat[i]) * factors
            prof = _profile_one_param(loss, loss_grad, hvp, x_hat, i, grid)
            results[f"profile_param_{i}"] = prof

            # compute confidence intervals from profiled likelihood
            prof_losses = prof["profile_loss"]
            LLH_best = res.fun
            threshold = LLH_best + 0.5 * 3.84  # 95% CI, chi2 with 1 dof

            def find_CI_side(prof_losses, grid, threshold, side="lower"):
                if side == "lower":
                    inds = np.arange(len(grid))[: len(grid) // 2][::-1]
                else:
                    inds = np.arange(len(grid))[len(grid) // 2 :]
                for ind in inds:
                    if prof_losses[ind] > threshold:
                        if ind == 0 or ind == len(grid) - 1:
                            return np.nan
                        else:
                            # linear interpolation for better accuracy
                            x0, x1 = grid[ind - 1], grid[ind]
                            y0, y1 = prof_losses[ind - 1], prof_losses[ind]
                            slope = (y1 - y0) / (x1 - x0)
                            return x0 + (threshold - y0) / slope
                return np.nan

            CIs.append(
                (
                    find_CI_side(prof_losses, grid, threshold, side="lower"),
                    find_CI_side(prof_losses, grid, threshold, side="upper"),
                )
            )
        results["CIs"] = np.array(CIs)  # shape (P, 2) lower, upper

    return results, fit_results, LLHs
