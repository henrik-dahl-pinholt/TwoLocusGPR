"""Gaussian process regression utilities built on JAX for MSD-based kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import minimize, root_scalar

from . import utils


@jax.jit
def LLH(data: jnp.ndarray, noise: jnp.ndarray, covmat: jnp.ndarray) -> jnp.ndarray:
    """Log-likelihood of a batch of single-track observations under a GP.

    Parameters
    ----------
    data : jnp.ndarray, shape (ndat, ndim)
        One track with optional NaNs for missing points.
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std.
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Prior covariance for each dimension at the observed times.

    Returns
    -------
    jnp.ndarray
        Scalar log-likelihood summed over dimensions and valid entries.
    """
    nan_entries, masked_data, mats_for_cholesky = utils.get_mat_for_cholesky(
        data, covmat, noise
    )
    cholesky_pred = jnp.linalg.cholesky(mats_for_cholesky)  # (ndim,ndat,ndat)

    z = jax.scipy.linalg.solve_triangular(
        cholesky_pred, masked_data.T, lower=True
    )  # (ndim,ndat)
    x = jax.scipy.linalg.solve_triangular(
        cholesky_pred.swapaxes(-1, -2), z, lower=False
    )  # (ndim,ndat)

    log_diag = jnp.linalg.diagonal(jnp.log(cholesky_pred))
    logdet = jnp.sum(log_diag, axis=-1)  # (ndim,)
    data_term = jnp.einsum("ji,ij->i", masked_data, x)  # (ndim,)
    llh = -0.5 * data_term - logdet  # (ndim,)

    return jnp.sum(llh) - jnp.sum(~nan_entries) * jnp.log(2 * jnp.pi) / 2


@jax.jit
def LLH_value_and_grad(data, noise, covmat, paramderivs):
    """Log-likelihood and gradient w.r.t. kernel parameters for one track.

    Parameters
    ----------
    data : jnp.ndarray, shape (ndat, ndim)
        One track with optional NaNs for missing points.
    noise : jnp.ndarray, shape (ndim,)
        Per-dimension observational noise std.
    covmat : jnp.ndarray, shape (ndim, ndat, ndat)
        Prior covariance for each dimension at the observed times.
    paramderivs : jnp.ndarray, shape (ndim, ndat, ndat, n_params)
        Covariance derivatives per parameter for each dimension.

    Returns
    -------
    llh : jnp.ndarray
        Scalar log-likelihood summed over dimensions and valid entries.
    grad : jnp.ndarray, shape (n_params,)
        Gradient of the log-likelihood with respect to parameters, with noise
        gradients stacked last.
    """
    n, d = data.shape

    nan_entries, y, mats_for_cholesky = utils.get_mat_for_cholesky(data, covmat, noise)

    # Cholesky factor for each dim
    L = jnp.linalg.cholesky(mats_for_cholesky)  # (d, n, n)

    z = jax.scipy.linalg.solve_triangular(L, y.T, lower=True)  # (d, n)
    alpha = jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L, -1, -2), z, lower=False
    )  # (d, n)

    # Log-determinant: log|K| = 2 * sum(log(diag(L)))
    logdet_per_dim = jnp.sum(
        jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1
    )  # (d,)
    # Data term: -1/2 y^T K^{-1} y = -1/2 sum_i alpha_i * y_i  (per dim)
    llh_data_terms = -0.5 * jnp.sum(alpha * y.T, axis=-1)  # (d,)

    # Combine per-dim terms; constant term only counts observed entries
    n_obs_total = jnp.sum(~nan_entries)  # scalar
    llh = jnp.sum(llh_data_terms - logdet_per_dim) - 0.5 * n_obs_total * jnp.log(
        2.0 * jnp.pi
    )

    # ===== Gradient =====
    # Rearrange paramderivs to (d, P, n, n) to align with dimensions
    dK = jnp.transpose(paramderivs, (0, 3, 1, 2))  # (d, P, n, n)

    # Term 1: 0.5 * alpha^T (dK/dtheta) alpha   -> shape (d, P)
    # Compute bilinear form without forming xx^T
    term1 = 0.5 * jnp.einsum("di,dpij,dj->dp", alpha, dK, alpha)

    # Term 2: 0.5 * tr(K^{-1} dK/dtheta)
    # Compute K^{-1} once per dim via two triangular solves with identity
    I = jnp.eye(n)
    tmp = jax.scipy.linalg.solve_triangular(
        L, jnp.broadcast_to(I, (d, n, n)), lower=True
    )  # (d, n, n)
    K_inv = jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L, -1, -2), tmp, lower=False
    )  # (d, n, n)

    # Frobenius inner product to get the trace for each (dim, param)
    # tr(K^{-1} dK) = sum_ij K_inv_{ij} * dK_{ji}
    term2 = 0.5 * jnp.einsum("dij,dpji->dp", K_inv, dK)

    # Sum over dims, result is (P,)
    grad = term1 - term2  # (d, P)
    noise_grad, paramgrad = grad[:, -1], grad[:, :-1]  # (d,1), (d, P-1)

    # flatten and concat to get final gradient
    grad = jnp.concatenate((paramgrad.flatten(), noise_grad))  # (P,)

    return llh, grad


vmap_LLH = jax.vmap(LLH, in_axes=(0, None, None))
vmap_value_and_grad = jax.vmap(LLH_value_and_grad, in_axes=(0, None, None, 0))


class GPR:
    """Gaussian Process regressor with MSD-parameterized covariance.

    Parameters
    ----------
    ts : array-like
        Observation times for training data.
    MSD_func : callable
        Function ``msd(t, params)`` returning the MSD at lag ``t``.
    dim : int
        Number of spatial dimensions in each track.

    Notes
    -----
    Parameter vectors are packed as ``[params_per_dim * dim, noise_per_dim]``
    where MSD parameters are tiled once per dimension followed by per-dim noise.
    NaNs in data are handled by inflating diagonal noise and masking values.
    """

    def __init__(self, ts, MSD_func, dim):
        self.ts = ts
        self.MSD_func = MSD_func
        self.dim = dim
        self._Construct_covbuilder()

    def _Construct_covbuilder(self):
        """Pre-build JIT-ed covariance and gradient builders for the MSD kernel."""
        msd_grad = jax.jacfwd(
            self.MSD_func, argnums=1
        )  # gradient of MSD with respect to parameters
        vmap_msd_grad = jax.vmap(
            msd_grad, in_axes=(None, 0)
        )  # vectorized gradient of MSD to run across dims
        vmap_msd = jax.vmap(
            self.MSD_func, in_axes=(None, 0)
        )  # vectorized MSD to run across dims

        @jax.jit
        def Build_covmat(theta, t1s, t2s):
            params_per_dim = theta.reshape(self.dim, -1)  # (ndim, nparams)

            covmat = 0.5 * (
                vmap_msd(1e23, params_per_dim)[:, None, None]
                - vmap_msd(t1s[:, None] - t2s[None, :], params_per_dim)
            )
            return covmat  # (ndim,ndat,ndat)

        @jax.jit
        def Build_covmat_grad(theta, data, t1s, t2s):
            ndim = data.shape[-1]
            # unpack parameters
            noise = theta[-ndim:]
            params = theta[:-ndim]
            params_per_dim = params.reshape(self.dim, -1)  # (ndim, nparams)
            param_grads = 0.5 * (
                vmap_msd_grad(1e23, params_per_dim)[:, None, None, :]
                - vmap_msd_grad(t1s[:, None] - t2s[None, :], params_per_dim)
            )  # (ndims,ndat,ndat,nparams)
            # covmat = 0.5 * (msd_grad(1e23,params_per_dim) - msd_grad(t1s[None,:, None,None] - t2s[None,None, :,None],params_per_dim[:,None,None,:])) # (ndims,ndat,ndat,nparams)

            # add noise
            nan_entries = jnp.isnan(data)  # (ndat,ndim)
            noise_vals = jnp.where(
                nan_entries, 0.0 * jnp.ones(len(noise)), 2 * noise
            )  # (ndat,ndim)

            diag_noise = jax.vmap(jnp.diag)(noise_vals.T)  # (ndim,ndat,ndat)

            # noise_grads = jnp.broadcast_to(diag_noise,(ndim,*covmat.shape) ) # (ndim,ndat,ndat,n_noise)
            # param_grads = jnp.broadcast_to(covmat,(ndim,*covmat.shape) )# (ndim,ndat,ndat,nparams)

            # merge the two gradients along the last axis
            mats_for_cholesky = jnp.concatenate(
                (param_grads, diag_noise[..., None]), axis=-1
            )  # (ndim,ndat,ndat,nparams+1)

            return mats_for_cholesky

        self.covbuilder = Build_covmat
        self.grad = jax.vmap(Build_covmat_grad, in_axes=(None, 0, None, None))

    def sample_prior(self, theta, num_samples, nanfrac=0.0, seed=42, batch_size=10):
        """Draw prior GP samples and optionally drop NaNs.

        Parameters
        ----------
        theta : array-like, shape (n_params * dim + dim,)
            Parameters tiled per-dimension followed by per-dimension noise.
        num_samples : int
            Number of tracks to sample.
        nanfrac : float, optional
            Fraction of entries to mask as NaN in each track.
        seed : int, optional
            PRNG seed.
        batch_size : int, optional
            Batch size for sampling to limit memory.
        """
        noise = theta[-self.dim :]  # last dim is noise
        paramvec = theta[: -self.dim]  # all but last dim are parameters

        key = jax.random.PRNGKey(seed)

        mu = jnp.zeros((num_samples, len(self.ts), self.dim))
        nan_mask = (
            jax.random.uniform(key, shape=(num_samples, len(self.ts), self.dim))
            < nanfrac
        )
        nan_mask = jnp.where(nan_mask, jnp.nan, 1.0)

        mu_iterator = mu * nan_mask

        covmat = self.covbuilder(paramvec, self.ts, self.ts)  # (ndim,ndat,ndat)

        if num_samples <= batch_size:
            batch_inds = [np.arange(num_samples)]
        else:
            batch_inds = np.array_split(
                np.arange(num_samples),
                num_samples // batch_size + 1,
            )
        sample_list = []

        seeds = (
            seed
            + jnp.arange(len(batch_inds))[:, None]
            + jnp.arange(self.dim)[None, :] * len(batch_inds)
        )

        for seedval, inds in zip(seeds, batch_inds):
            mus = mu_iterator[inds]

            samples = jax.vmap(utils.sample_gauss_cmat, in_axes=(2, 0, 0))(
                mus,
                covmat,
                seedval,
            ).transpose(1, 2, 0)

            sample_list.append(samples)
        samples = np.concatenate(sample_list, axis=0)

        # output = [samples]

        key, subkey_noise, subkey_nan = jax.random.split(key, 3)

        noisy_samples = samples + np.sqrt(2) * noise[None, None, :] * jax.random.normal(
            subkey_noise, shape=samples.shape
        )

        return samples, noisy_samples

    def Predict(
        self,
        x,
        prediction_points,
        noisy_samples,
        batch_size=100,
        verbose=True,
    ):
        """Posterior mean/variance at prediction points for many tracks.

        Parameters
        ----------
        x : array-like
            Parameter vector in log-abs space (will be abs-ed internally).
        prediction_points : array-like
            Times at which to predict.
        noisy_samples : array-like, shape (n_tracks, n_time, dim)
            Observed tracks (can include NaNs).
        batch_size : int, optional
            Batch size for prediction loops.
        verbose : bool, optional
            Show progress bar when True.
        """
        noise = jnp.abs(x[-self.dim :])  # last dim is noise
        params = jnp.abs(x[: -self.dim])  # all but last dim are parameters
        final_params = jnp.concatenate((jnp.tile(params, self.dim), noise))
        theta = final_params[: -self.dim]  # all but last dim are parameters

        prediction_covmat = self.covbuilder(theta, prediction_points, prediction_points)
        prediction_kernel = self.covbuilder(theta, prediction_points, self.ts)
        covmat = self.covbuilder(theta, self.ts, self.ts)

        mean_out = np.zeros((len(noisy_samples), len(prediction_points), self.dim))
        var_out = np.zeros((len(noisy_samples), len(prediction_points), self.dim))

        data_batches = np.array_split(
            noisy_samples, len(noisy_samples) // batch_size + 1
        )
        if verbose:
            data_batches = tqdm(
                data_batches, desc="Predicting", total=len(data_batches)
            )
        else:
            data_batches = iter(data_batches)
        count = 0
        for batch in data_batches:
            mean_pred = utils.v_pred(prediction_kernel, batch, covmat, noise)
            # print diagnostics for numerical stability of the covariance
            print("")

            cov_pred = utils.vmap_pred_cov(
                prediction_kernel, covmat, batch, prediction_covmat, noise
            )
            mean_out[count : count + len(batch)] = mean_pred
            var_out[count : count + len(batch)] = cov_pred
            count += len(batch)
        # mean_pred = v_pred(prediction_kernel, noisy_samples, covmat,noise)
        # cov_pred = vmap_pred_cov(prediction_kernel, covmat, noisy_samples, prediction_covmat, noise)

        # return mean_pred,cov_pred
        return mean_out, var_out

    def sample_posterior(
        self,
        x,
        prediction_points,
        noisy_samples,
        num_samples=100,
        seed=42,
        verbose=True,
        batch_size=100,
    ):
        """Draw posterior GP samples at prediction points given observed tracks.

        Parameters
        ----------
        x : array-like
            Parameter vector in log-abs space (will be abs-ed internally).
        prediction_points : array-like
            Times at which to sample the posterior.
        noisy_samples : array-like, shape (n_tracks, n_time, dim)
            Observed tracks (can include NaNs).
        num_samples : int, optional
            Number of posterior samples per track.
        seed : int, optional
            PRNG seed for sampling.
        verbose : bool, optional
            Show progress bar when True.
        batch_size : int, optional
            Batch size for prediction loops.
        """
        noise = jnp.abs(x[-self.dim :])  # last dim is noise
        params = jnp.abs(x[: -self.dim])  # all but last dim are parameters
        final_params = jnp.concatenate((jnp.tile(params, self.dim), noise))
        theta = final_params[: -self.dim]  # all but last dim are parameters

        prediction_covmat = self.covbuilder(theta, prediction_points, prediction_points)
        prediction_kernel = self.covbuilder(theta, prediction_points, self.ts)
        covmat = self.covbuilder(theta, self.ts, self.ts)  # (ndim,ndat,ndat)

        out_samps = np.zeros(
            (len(noisy_samples), num_samples, len(prediction_points), self.dim)
        )
        data_batches = np.array_split(
            noisy_samples, len(noisy_samples) // batch_size + 1
        )
        if verbose:
            iterator = tqdm(
                data_batches, desc="Sampling posterior", total=len(data_batches)
            )
        else:
            iterator = data_batches
        count = 0
        # for i in iterator:
        for batch in iterator:
            # mean_pred = predict_mean_single(prediction_kernel, noisy_samples[i], covmat, noise)
            # cov_pred = predict_cov_single(prediction_kernel, covmat, noisy_samples[i], prediction_covmat, noise)
            # out_samps[i] = sample_gauss(num_samples, mean_pred, cov_pred, seed=seed+i)
            mean_pred = utils.v_pred(prediction_kernel, batch, covmat, noise)
            cov_pred = utils.vmap_pred_cov_full(
                prediction_kernel, covmat, batch, prediction_covmat, noise
            )

            out_samps[count : count + len(batch)] = jax.vmap(
                utils.sample_gauss, in_axes=(None, 0, 0, None)
            )(num_samples, mean_pred, cov_pred.transpose(0, 3, 1, 2), seed + count)
            count += len(batch)
        return out_samps

    def LLH(self, theta, data, batch_size=500, verbose=False):
        """Total log-likelihood across a batch of tracks."""
        noise = theta[-self.dim :]  # last dim is noise
        paramvec = theta[: -self.dim]  # all but last dim are parameters

        covmat = self.covbuilder(paramvec, self.ts, self.ts)

        data_batches = np.array_split(data, len(data) // batch_size + 1)
        llhs = 0
        if verbose:
            data_batches = tqdm(
                data_batches, desc="Computing LLH", total=len(data_batches)
            )
        else:
            data_batches = iter(data_batches)
        for batch in data_batches:
            llhs += jnp.sum(vmap_LLH(batch, noise, covmat))

        return llhs

    def get_objective(self, data):
        """Return a JIT-ed log-likelihood closure in log-parameter space."""

        @jax.jit
        def objective(x):

            noise = jnp.exp(x[-self.dim :])  # last dim is noise
            params = jnp.exp(x[: -self.dim])  # all but last dim are parameters
            final_params = jnp.concatenate((jnp.tile(params, self.dim), noise))
            llh = self.LLH(final_params, data)

            return llh

        return objective

    def MCMC(
        self,
        data,
        initial_guess,
        step_sizes,
        verbose=True,
        seed=42,
        n_samples=1000,
        fixed_params=[],
    ):
        """Simple Metropolis-Hastings sampler over log-parameters.

        Parameters
        ----------
        data : jnp.ndarray, shape (n_tracks, n_time, dim)
            Observed tracks used to evaluate the likelihood.
        initial_guess : array-like
            Starting point in log-parameter space.
        step_sizes : array-like
            Proposal scales in log space.
        verbose : bool, optional
            Show sampling progress when True.
        seed : int, optional
            PRNG seed for reproducibility.
        n_samples : int, optional
            Number of MCMC samples to draw.
        fixed_params : list[int], optional
            Indices to freeze (step size set to zero).

        Returns
        -------
        results : jnp.ndarray, shape (n_samples, n_params)
            Sampled parameter states.
        llhs : list[float]
            Log-likelihood trace corresponding to samples.
        """

        objective = self.get_objective(data)
        freeze_mask = np.ones_like(initial_guess)
        for i in fixed_params:
            freeze_mask[i] = 0.0
        step_sizes = step_sizes * freeze_mask

        @jax.jit
        def do_step(rng_key, current_position, objective_current):
            proposal = (
                current_position
                + jax.random.normal(rng_key, shape=current_position.shape) * step_sizes
            )
            objective_proposal = objective(proposal)
            accept_prob = jnp.exp(objective_proposal - objective_current)
            rng_key, subkey = jax.random.split(rng_key)
            u = jax.random.uniform(subkey)
            accept = u < accept_prob
            new_position = jnp.where(accept, proposal, current_position)
            new_objective = jnp.where(accept, objective_proposal, objective_current)
            return new_position, new_objective

        rng_key = jax.random.PRNGKey(seed)
        results = []
        llhs = []
        state_position = initial_guess
        state_objective = objective(initial_guess)
        if verbose:
            iterator = tqdm(
                range(n_samples),
                desc="Sampling with MCMC",
                total=n_samples,
                leave=False,
            )
        else:
            iterator = range(n_samples)
        for i in iterator:
            rng_key, sample_key = jax.random.split(rng_key)
            state_position, state_objective = do_step(
                sample_key, state_position, state_objective
            )
            results.append(state_position)
            llhs.append(-state_objective)
        results = jnp.array(results)
        return results, llhs
