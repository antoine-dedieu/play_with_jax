# Compare Gibbs sampling in numba vs JAX for an Ising model on a 2D lattice

import numba as nb
import numpy as np
import time
from jax.scipy.special import logsumexp
from functools import partial

from jax import jit, random
from jax.lax import dynamic_slice, dynamic_update_slice, scan
from jax.nn import sigmoid
from jax import numpy as jnp

#####################################
############## Ising model ##########
#####################################

def ising_matrix(grid_side, rho=1):
    # Ising model on a 2D periodic lattice
    W = np.zeros((grid_side ** 2, grid_side **2))

    for i in range(grid_side):
        for j in range(grid_side):
            idx = i * grid_side + j
            idx_up = (i - 1) % grid_side * grid_side + j
            idx_down = (i + 1) % grid_side * grid_side + j
            idx_left = i * grid_side + (j - 1) % grid_side
            idx_right = i * grid_side + (j + 1) % grid_side

            for idx_neighbor in [idx_up, idx_down, idx_left, idx_right]:
                W[idx, idx_neighbor] = rho
    return W


def logZ_estimate(S):
    # Ogata Tanamura estimator of the log-partition function
    # http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/NormConstants/PotamianosGoutsiasIEEE1997.pdf
    n_samples, d = S.shape
    energy = - ((S @ W) * S).sum(1)
    logZ = -logsumexp(energy) + d * np.log(2) + np.log(n_samples)
    return float(logZ)


#####################################
###### Gibbs sampling in numba ######
#####################################

@nb.vectorize()
def nb_sigmoid(x):
    if x > 0:
        x = np.exp(x)
        return x / (1 + x)
    else:
        return 1 / (1 + np.exp(-x))

@nb.njit(cache=True)
def gibbs_ising_numba(W, n_samples, n_steps=1000):
    # The result is reflected in S, which is updated in place
    d = W.shape[0]
    S = 2 * (np.random.rand(n_samples, d) < 0.5).astype(np.float64) - 1

    assert W.shape == (d, d)
    assert (np.diag(W) == 0).all()
    assert (W == W.T).all()

    g = S.dot(W.T)  # size N_samples x d, g_ij = x^{(i)}^T w_j
    for step in range(n_steps):
        for j in np.random.permutation(d):
            delta = -2 * g[:, j : j + 1] * S[:, j : j + 1]
            threshold = nb_sigmoid(delta)  # p(switch_j | x_{-j}) = sigmoid(- 2 * x_j * g_j)
            flip = (np.random.rand(n_samples, 1) < threshold).astype(np.float64)

            # Update S
            S[:, j : j + 1] = (1 - 2 * flip) * S[:, j :j + 1]

            # Update g
            g += flip * 2 * S[:, j : j + 1] * W[j :j + 1]
    return S


#####################################
####### Gibbs sampling in JAX #######
#####################################

@jit
def update_gibbs_j(gSrng, j):
    g, S, rng = gSrng
    n_samples, d = S.shape

    # g = S @ W.T + b.T
    S_j = dynamic_slice(S, (0, j), (n_samples, 1))
    g_j = dynamic_slice(g, (0, j), (n_samples, 1))

    # Update S
    delta = - 2 * S_j * g_j
    threshold = sigmoid(delta)
    rng, rng_input = random.split(rng)
    flip = random.bernoulli(rng_input, p=threshold, shape=(n_samples, 1))
    S = dynamic_update_slice(S, (1 - 2 * flip) * S_j, (0, j))

    # Update g
    S_j = dynamic_slice(S, (0, j), (n_samples, 1))
    W_j = dynamic_slice(W, (j, 0), (1, d))
    delta_g = flip * 2 * S_j * W_j
    g = g.at[:].add(delta_g)
    return (g, S, rng), None

@jit
def update_gibbs(gSrng, _):
    g, S, rng = gSrng
    n_samples, d = S.shape
    rng, rng_input = random.split(rng)
    order = random.permutation(rng_input, d)
#     for j in order:
#         (g, S, rng), _ = update_gibbs_j((g, S, rng), j)
    g, S, rng = scan(update_gibbs_j, (g, S, rng), order)[0]
    return (g, S, rng), None


@partial(jit, static_argnums=(1, 2))  # jit with axis being static
def gibbs_ising_jax(W, n_samples, n_steps=1000, rng=random.PRNGKey(42)):
    # Vectorization of Gibbs sampling for Ising model
    d = W.shape[0]
    rng, rng_input = random.split(rng)
    S = 2 * random.bernoulli(rng_input, p=0.5, shape=(n_samples, d)).astype(np.float32) - 1

    g = S @ W.T
    iters = jnp.arange(n_steps)
#     for it in iters:
#         (g, S, rng), _ = update_gibbs((g, S, rng), it)
    g, S, rng = scan(update_gibbs, (g, S, rng), iters)[0]
    return S


if __name__=="__main__":
    grid_side = 5
    n_samples = 1000
    n_steps = 1000

    # Simulate Ising model
    W = ising_matrix(grid_side)

    # Run both methods twice to use jitted code
    # Numba
    _ = gibbs_ising_numba(W, n_samples=n_samples, n_steps=n_steps)
    start = time.time()
    S1 = gibbs_ising_numba(W, n_samples=n_samples, n_steps=n_steps)
    t_numba = time.time() - start
    print(f"Sampling time with numba: {t_numba:.3f}s")

    # JAX
    _ = gibbs_ising_jax(W, n_samples=n_samples, n_steps=n_steps)
    start = time.time()
    S2 = gibbs_ising_jax(W, n_samples=n_samples, n_steps=n_steps)
    S2.block_until_ready()
    t_jax = time.time() - start
    print(f"Sampling time with JAX: {t_jax:.3f}s")
    print(f"JAX speed up: {t_numba / t_jax:.3f}x")

    # Check that both methods give a similar estimate of the log-partition function
    print(f"\nNumba log-partition function estimate: {logZ_estimate(S1):.2f}")
    print(f"JAX log-partition function estimate: {logZ_estimate(S2):.2f}")
