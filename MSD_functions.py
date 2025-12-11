"""MSD model components used by the GP kernels (JAX-friendly)."""

import jax
import jax.numpy as jnp

@jax.jit
def softmin(x: jnp.ndarray, axis=-1) -> jnp.ndarray:
    """ Compute the softmin of an array along a specified axis.

    Parameters
    ----------
    x : jnp.ndarray, shape (..., n, ...)
        Input array.
    axis : int, optional
        Axis along which to compute the softmin, by default -1

    Returns
    -------
    jnp.ndarray
        Softmin of the input array along the specified axis.
    """
    x_max = jnp.max(-x, axis=axis, keepdims=True)
    exps = jnp.exp(-(x + x_max))  # equivalent to exp(-x - max(-x))
    return exps / jnp.sum(exps, axis=axis, keepdims=True)


@jax.jit
def Rouse_MSD(t: jnp.ndarray, params: tuple) -> jnp.ndarray:
    """ Compute the Rouse model mean squared displacement (MSD) at time t.

    Parameters
    ----------
    t : jnp.ndarray
        Time points at which to compute the MSD.
    params : tuple
        Model parameters (Gamma, J).

    Returns
    -------
    jnp.ndarray
        MSD values at the given time points.
    """
    Gamma, J = params
    abs_t = jnp.abs(t)
    tau = (J / Gamma) ** 2 / jnp.pi
    return 2 * Gamma * jnp.sqrt(abs_t) * (
        1 - jnp.exp(-tau / (abs_t + 1e-8))
    ) + 2 * J * jax.scipy.special.erfc(jnp.sqrt(tau / (abs_t + 1e-8)))


@jax.jit
def Saturating_MSD(t: jnp.ndarray, params: tuple, eps: float=1e-8, n: int=2) -> jnp.ndarray:
    """ Compute the saturating mean squared displacement (MSD) at time t.

    Parameters
    ----------
    t : jnp.ndarray
        Time points at which to compute the MSD.
        
    params : tuple
        Model parameters (Gamma, J, alpha).
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-8
    n : int, optional
        Sharpness parameter for the soft minimum, by default 2
    Returns
    -------
    jnp.ndarray
        MSD values at the given time points.
    """
    Gamma, J, alpha = params
    safe_t = jnp.sqrt(t**2 + eps**2)
    log_short_time = jnp.log(2 * Gamma) + alpha * jnp.log(safe_t)
    log_long_time = jnp.log(2 * J) * jnp.ones_like(log_short_time)
    return jnp.exp(
        -(1 / n)
        * jax.scipy.special.logsumexp(
            (-n) * jnp.array([log_short_time, log_long_time]), axis=0
        )
    )