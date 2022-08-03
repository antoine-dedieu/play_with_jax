# Compare CPU vs numba vs JAX for a simple task of filling in a matrix

import numba as nb
import numpy as np
import time
import jax
import jax.numpy as jnp

def f_cpu(N, P):
    W = np.zeros((N, P))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = i ** 2 + j ** 3
    return W


@nb.njit(cache=True)
def f_numba(N, P):
    W = np.zeros((N, P))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = i ** 2 + j ** 3
    return W


def f_jax_ij(i, j):
    return i ** 2 + j ** 3

def f_jax(N, P): 
    f_jax_vmap = jax.vmap(jax.vmap(f_jax_ij, in_axes=(None, 0)), in_axes=(0, None))
    return f_jax_vmap(jnp.arange(N), jnp.arange(P))


if __name__=="__main__":
    N = 10000
    P = 10000

    # CPU
    start = time.time()
    W_cpu = f_cpu(N, P)
    t_cpu = time.time() - start
    print(f"CPU time: {t_cpu:.3f}s")

    # Run numba and JAX twice to use jitted code
    # Numba
    _ = f_numba(N, P)
    start = time.time()
    W_numba = f_numba(N, P)
    t_numba = time.time() - start
    print(f"Numba time: {t_numba:.3f}s")

    # JAX
    _ = f_jax(N, P)
    start = time.time()
    W_jax = f_jax(N, P)
    W_jax.block_until_ready()
    t_jax = time.time() - start
    print(f"JAX time: {t_jax:.3f}s")

    print(f"\nNumba / CPU speed up: {t_cpu / t_numba:.3f}x")
    print(f"JAX / numba speed up: {t_numba / t_jax:.3f}x")

    assert np.abs(W_cpu - W_numba).max() < 1e-6
    assert np.abs(W_cpu - W_jax).max() < 1e-6
