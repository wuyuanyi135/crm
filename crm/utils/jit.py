from typing import List

from numba import jit, prange
import numpy as np


@jit(nopython=True, cache=True)
def per_particle_volume_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    """
    ignore the count, calculate the per-particle volume
    :param n:
    :return:
    """
    out = np.ones((n.shape[0],))
    for i in range(n.shape[1] - 1):
        out *= n[:, i] ** volume_fraction_powers[i]
    return out * shape_factor


@jit(nopython=True, cache=True)
def particle_volume_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    return per_particle_volume_jit(n, volume_fraction_powers, shape_factor) * n[:, -1]


@jit(nopython=True, cache=True)
def volume_fraction_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    return particle_volume_jit(n, volume_fraction_powers, shape_factor).sum(axis=0)


@jit(nopython=True, cache=True)
def volume_average_size_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float,
                            per_particle=False):
    """

    :param n:
    :param volume_fraction_powers:
    :param shape_factor:
    :param per_particle: ignore count, compute the size when all particles in n agglomerating into one particle
    :return:
    """
    nrows = n.shape[0]
    ncols = n.shape[1]
    if nrows == 0:
        return np.zeros((n.shape[1],))
    ret = np.empty((ncols,))

    if per_particle:
        count = 1
        particle_volume = per_particle_volume_jit(n, volume_fraction_powers, shape_factor).sum()
        particle_average_volume = particle_volume
    else:
        count = n[:, -1].sum()
        particle_average_volume = volume_fraction_jit(n, volume_fraction_powers, shape_factor) / count

    if volume_fraction_powers.size == 1:
        # one dimensional
        ret[0] = (particle_average_volume / shape_factor) ** (1 / volume_fraction_powers[0])
        ret[1] = count
        return ret
    else:
        # multi-dimensional N
        non_first_dim_mean_sizes = n[:, 1:-1].sum(axis=0) / nrows
        particle_average_volume = particle_average_volume / shape_factor
        non_first_dim_prod = np.prod(non_first_dim_mean_sizes ** volume_fraction_powers[1:])

        # exclude the effect of the non first dimensions. The modified particle_average_volume can be used to
        # calculate the first dimension by reciprocal power of the first dimension
        particle_average_volume = particle_average_volume / non_first_dim_prod

        first_dim_size = particle_average_volume ** (1 / volume_fraction_powers[0])

        ret[0] = first_dim_size
        ret[1:-1] = non_first_dim_prod
        ret[-1] = count

        return ret


@jit(nopython=True, cache=True)
def partition_equivalent_rows_jit(ns, volume_fraction_powers: np.ndarray, shape_factor: float) -> np.ndarray:
    ncols = ns[0].shape[1]
    nrows = len(ns)
    ret = np.zeros((nrows, ncols))

    for i in prange(nrows):
        p = ns[i]
        if p.size == 0:
            continue
        equivalent_row = volume_average_size_jit(p, volume_fraction_powers, shape_factor)
        ret[i, :] = equivalent_row[0, :]

    ret = ret[ret[:, -1] != 0, :]
    return ret


@jit(nopython=True, cache=True)
def binary_agglomeration_jit(n, alpha: float, volume_fraction_powers: np.ndarray, shape_factor: float):
    """
    Constant binary agglomeration
    TODO: implement agglomeration along different dimension
    :param ns:
    :param alpha_poly:
    :return: B, D. B is the same form as n. D is a vector of count change rate
    """
    nrows = n.shape[0]
    ncols = n.shape[1]

    combination_index = np.triu_indices(nrows, 1)
    # (n choose 2, 2, dim+1) second: two rows for combination
    # combination_table = n[combination_index, :].swapaxes(0, 1)
    # ND indexing is not supported in numba.
    combination_table = np.empty((combination_index[0].shape[0], 2, ncols))

    for i, (self_row_id, other_row_id) in enumerate(zip(*combination_index)):
        combination_table[i, 0, :] = n[self_row_id, :]
        combination_table[i, 1, :] = n[other_row_id, :]

    D = np.zeros((nrows, ))

    # shape: (n choose 2, dim + 1)
    B = np.empty((combination_table.shape[0], ncols))

    for i, row_idx in enumerate(zip(*combination_index)):
        agglomeration_parent_rows = combination_table[i]

        reduced = 1 / 2 * alpha * np.prod(agglomeration_parent_rows[:, -1])

        # this for loop should just run twice for binary agglomeration.
        for r in row_idx:
            D[r] += reduced

        B[i, :] = volume_average_size_jit(agglomeration_parent_rows, volume_fraction_powers, shape_factor,
                                          per_particle=True)
        B[i, -1] = reduced
    return B, D
