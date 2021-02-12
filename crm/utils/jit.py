from typing import List

from numba import jit, prange, typed
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


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
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


@jit(nopython=True, cache=True, nogil=True)
def binary_agglomeration_jit(n, alpha: float, volume_fraction_powers: np.ndarray, shape_factor: float,
                             crystallizer_volume: float,
                             combination_index_=None, combination_table_=None, compression_interval: float = 0.):
    """
    Constant binary agglomeration
    TODO: implement agglomeration along different dimension
    :param compression_interval: if 0 not compress. otherwise compress with this interval.
    :param combination_index_:
    :param ns:
    :param alpha_poly:
    :return: B, D. B is the same form as n. D is a vector of count change rate
    """
    nrows = n.shape[0]
    ncols = n.shape[1]

    crystallizer_volume_square = crystallizer_volume ** 2
    if combination_index_ is None:
        combination_index = np.triu_indices(nrows, 0)  # combination can self-intersect!
    else:
        combination_index = combination_index_
    # (n choose 2, 2, dim+1) second: two rows for combination
    # combination_table = n[combination_index, :].swapaxes(0, 1)
    # ND indexing is not supported in numba.
    if combination_table_ is None:
        combination_table: np.ndarray = np.empty((combination_index[0].shape[0], 2, ncols))

        for i, (self_row_id, other_row_id) in enumerate(zip(*combination_index)):
            combination_table[i, 0, :] = n[self_row_id, :]
            combination_table[i, 1, :] = n[other_row_id, :]
    else:
        combination_table = combination_table_

    D = np.zeros((nrows,))

    # shape: (n choose 2, dim + 1)
    B = np.empty((combination_table.shape[0], ncols))

    for i, row_idx in enumerate(zip(*combination_index)):
        agglomeration_parent_rows = combination_table[i]

        reduced = 1 / 2 * alpha * np.prod(agglomeration_parent_rows[:, -1]) * crystallizer_volume_square

        # this for loop should just run twice for binary agglomeration.
        for r in row_idx:
            D[r] += reduced

        B[i, :] = volume_average_size_jit(agglomeration_parent_rows, volume_fraction_powers, shape_factor,
                                          per_particle=True)
        B[i, -1] = reduced

    if compression_interval > 0:
        B = compress_jit(B, volume_fraction_powers, shape_factor, compression_interval)
    return B, D


def binary_agglomeration_parallel_wrapper(n, alpha: float, volume_fraction_powers: np.ndarray, shape_factor: float):
    import threading
    import os
    ncpu = os.cpu_count()

    nrows = n.shape[0]
    per_cpu_rows = nrows / ncpu

    combination_index = np.triu_indices(nrows, 1)
    combination_table = n[combination_index, :].swapaxes(0, 1)

    # trunk index
    sub_index = []
    sub_table = []
    for i in range(0, len(combination_index[0]), per_cpu_rows):
        sub_index.append((
            combination_index[0][i: i + per_cpu_rows],
            combination_index[1][i: i + per_cpu_rows]
        ))
        sub_table.append(combination_table[i: i + per_cpu_rows, :])


@jit(nopython=True, cache=True)
def compress_jit(n, volume_fraction_powers: np.ndarray, shape_factor: float, interval: float = 1e-6):
    """
    accelerated compression algorithm
    TODO choose which dimension to adjust (adjust dim)
    :param n:
    :param interval:
    :return:
    """
    adjust_dim = 0
    ndims = n.shape[1] - 1

    dims_min = np.empty((ndims,))
    dims_max = np.empty((ndims,))

    # store the group's count and volume in grid, so that no list of intermediate variable-size array will be created
    for i in range(ndims):
        col = n[:, i]
        dims_min[i] = col.min()
        dims_max[i] = col.max()
    cnts = ((dims_max - dims_min) // interval + 1).astype(np.int_)
    cumprod_dim = np.cumprod(np.hstack((np.array([1]), cnts)))[:-1]
    # nd grid is flattened into 1d
    # the first dimension is the slowest dimension.
    volume_grid = np.zeros((np.prod(cnts),))
    count_grid = np.zeros((np.prod(cnts),))
    sum_non_adjust_dim_grid = np.zeros((np.prod(cnts), ndims - 1))

    n_non_adjust_dim = np.delete(n.T, (adjust_dim, -1)).T

    for i, (row, non_adjust_row) in enumerate(zip(n, n_non_adjust_dim)):
        sizes = row[:-1]

        # mapping sizes to indices
        idx = (sizes - dims_min) // interval

        # convert 2d index to 1d index
        flat_idx = int((cumprod_dim * idx).sum())

        vol = volume_fraction_jit(np.expand_dims(row, 0), volume_fraction_powers, shape_factor)
        volume_grid[flat_idx] += vol
        count_grid[flat_idx] += row[-1]

        sum_non_adjust_dim_grid[flat_idx, :] += non_adjust_row

    # remove the empty rows in the grid
    empty_idx = count_grid == 0
    volume_grid = volume_grid[~empty_idx]
    count_grid = count_grid[~empty_idx]
    sum_non_adjust_dim_grid = sum_non_adjust_dim_grid[~empty_idx]

    # compute the equivalent size
    if sum_non_adjust_dim_grid.size == 0:
        # 1d
        adjust_dim_size = (volume_grid / count_grid / shape_factor) ** (1 / volume_fraction_powers)
        return np.hstack((adjust_dim_size.reshape((-1, 1)), count_grid.reshape((-1, 1))))
    else:
        mean_non_adjust_dim_grid = sum_non_adjust_dim_grid / count_grid.reshape((-1, 1))
        non_adjust_power = np.delete(volume_fraction_powers, adjust_dim)
        non_adjust_dim_vol_component = mean_non_adjust_dim_grid ** non_adjust_power

        # product along rows to compute the volume from the component
        non_adjust_dim_vol = np.ones((non_adjust_dim_vol_component.shape[0],))
        for _, components in enumerate(non_adjust_dim_vol_component.T):
            non_adjust_dim_vol *= components

        adjust_dim_size = (volume_grid / count_grid / shape_factor / non_adjust_dim_vol) ** (1 / volume_fraction_powers[
            adjust_dim])

        # TODO: this step does not support adjust_dim
        ret = np.hstack((adjust_dim_size.reshape((-1, 1)), mean_non_adjust_dim_grid, count_grid.reshape((-1, 1))))
        return ret
