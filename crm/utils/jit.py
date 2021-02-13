import os
import threading
from typing import List

from numba import jit, prange, typed
import numpy as np


@jit(nopython=True, cache=True, nogil=True)
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


@jit(nopython=True, cache=True, nogil=True)
def particle_volume_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    return per_particle_volume_jit(n, volume_fraction_powers, shape_factor) * n[:, -1]


@jit(nopython=True, cache=True, nogil=True)
def volume_fraction_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    return particle_volume_jit(n, volume_fraction_powers, shape_factor).sum(axis=0)


@jit(nopython=True, cache=True, nogil=True)
def volume_average_size_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float, mode: float = 0.):
    """

    :param n:
    :param volume_fraction_powers:
    :param shape_factor:
    :param mode: internal use only. mode 0: normal, <0: agglomeration, >0: breakage, treated as split ratio.
    :return:
    """
    nrows = n.shape[0]
    ncols = n.shape[1]
    if nrows == 0:
        return np.zeros((n.shape[1],))
    ret = np.empty((ncols,))

    if mode == 0.:
        count = n[:, -1].sum()
        particle_average_volume = volume_fraction_jit(n, volume_fraction_powers, shape_factor) / count
    elif mode < 0.:
        count = 0  # doesn't care
        particle_average_volume = per_particle_volume_jit(n, volume_fraction_powers, shape_factor).sum()
    else:
        count = 0  # doesn't care
        particle_average_volume = mode * per_particle_volume_jit(n, volume_fraction_powers, shape_factor).sum()

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
def binary_agglomeration_jit(
        n: np.ndarray,
        alpha: float,
        volume_fraction_powers: np.ndarray,
        shape_factor: float,
        crystallizer_volume: float,
        compression_interval: float = 0.,
        minimum_count: float = 1000.
):
    """
    Constant binary agglomeration
    TODO: implement agglomeration along different dimension
    :param minimum_count: when one of the involved class has lower than minimum_count particles, agglomeration is ignored.
    :param compression_interval: if 0 not compress. otherwise compress with this interval.
    :param n:
    :param alpha:
    :return: B, D. B is the same form as n. D is a vector of count change rate
    """

    # filter out the ignored particles
    included_rows = n[:, -1] >= minimum_count
    n_original = n
    n = n[included_rows]

    nrows = n.shape[0]
    ncols = n.shape[1]
    crystallizer_volume_square = crystallizer_volume ** 2
    combination_index = np.triu_indices(nrows, 0)  # combination can self-intersect!

    # (n choose 2, 2, dim+1) second: two rows for combination
    # combination_table = n[combination_index, :].swapaxes(0, 1)
    # ND indexing is not supported in numba.
    combination_table: np.ndarray = np.empty((combination_index[0].shape[0], 2, ncols))

    for i, (self_row_id, other_row_id) in enumerate(zip(*combination_index)):
        combination_table[i, 0, :] = n[self_row_id, :]
        combination_table[i, 1, :] = n[other_row_id, :]

    B, D = binary_agglomeration_internal(alpha, combination_index, combination_table, crystallizer_volume_square, ncols,
                                         nrows,
                                         shape_factor, volume_fraction_powers)

    if compression_interval > 0:
        B = compress_jit(B, volume_fraction_powers, shape_factor, compression_interval)

    # restore D to the same rows as original n
    D_original = np.zeros((n_original.shape[0],))
    D_original[included_rows] = D

    return B, D_original


@jit(nopython=True, cache=True, nogil=True)
def binary_agglomeration_internal(
        alpha,
        combination_index,
        combination_table,
        crystallizer_volume_square,
        ncols,
        nrows,
        shape_factor,
        volume_fraction_powers
):
    D = np.zeros((nrows,))
    # shape: (n choose 2, dim + 1)
    B = np.empty((combination_table.shape[0], ncols))
    for i, row_idx in enumerate(zip(*combination_index)):
        agglomeration_parent_rows = combination_table[i]

        reduced = alpha * np.prod(agglomeration_parent_rows[:, -1]) * crystallizer_volume_square

        # this for loop should just run twice for binary agglomeration.
        for r in row_idx:
            D[r] += reduced

        B[i, :] = volume_average_size_jit(agglomeration_parent_rows, volume_fraction_powers, shape_factor, mode=-1.)
        B[i, -1] = reduced
    return B, D


def binary_agglomeration_multithread(
        n: np.ndarray,
        alpha: float,
        volume_fraction_powers: np.ndarray,
        shape_factor: float,
        crystallizer_volume: float,
        compression_interval: float = 0.,
        minimum_count: float = 1000.,
        nthread=None,
):
    nthread = nthread or os.cpu_count()

    included_rows = n[:, -1] >= minimum_count
    n_original = n
    n = n[included_rows]

    nrows = n.shape[0]
    ncols = n.shape[1]
    crystallizer_volume_square = crystallizer_volume ** 2
    combination_index = np.triu_indices(nrows, 0)
    combination_table = n[combination_index, :].swapaxes(0, 1)

    # split into multiple arrays
    idx = np.arange(combination_table.shape[0], dtype=np.int)
    subarray_idx = np.array_split(idx, nthread)
    index_subarray = [(combination_index[0][x], combination_index[1][x]) for x in subarray_idx]
    table_subarray = [combination_table[x] for x in subarray_idx]
    x = zip(index_subarray, table_subarray)

    D = []
    B = []
    def map_func(x):
        idx, tbl = x
        B_, D_ = binary_agglomeration_internal(alpha, idx, tbl, crystallizer_volume_square, ncols,
                                             nrows,
                                             shape_factor, volume_fraction_powers)
        D.append(D_)
        B.append(B_)

    ths = [threading.Thread(target=map_func, args=(arg, )) for arg in x]
    for th in ths:
        th.start()
    for th in ths:
        th.join()


    D = sum(D)
    B = np.vstack(B)

    if compression_interval > 0:
        B = compress_jit(B, volume_fraction_powers, shape_factor, compression_interval)

    # restore D to the same rows as original n
    D_original = np.zeros((n_original.shape[0],))
    D_original[included_rows] = D

    return B, D_original


@jit(nopython=True, cache=True, nogil=True)
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


@jit(nopython=True, cache=True, nogil=True)
def binary_breakage_jit(
        n: np.ndarray,
        kernels: np.ndarray,
        volume_fraction_powers, shape_factor,
        crystallizer_volume: float,
        compression_interval: float = 0,
        minimum_count: float = 1000
):
    """
    :param n:
    :param crystallizer_volume:
    :param compression_interval:
    :param minimum_count:
    :param kernels: N x 2 array. N is the combinations. The two columns are split ratio and kernel value
    :param volume_fraction_powers:
    :param shape_factor:
    :return:
    """
    # filter out the ignored particles
    included_rows = n[:, -1] >= minimum_count
    n_original = n
    n = n[included_rows]

    nrows = n.shape[0]
    nkernels = kernels.shape[0]
    ncols = n.shape[1]
    crystallizer_volume_square = crystallizer_volume ** 2

    D = np.zeros((nrows,))

    # shape: (n * kernels * 2, dim + 1)
    # when evenly break, the two rows will be same.
    B = np.empty((nrows * nkernels * 2, ncols))

    B_index = 0
    for i, r in enumerate(n):
        for _, kernel in enumerate(kernels):
            split_ratio = kernel[0]
            k = kernel[1]

            reduced = k * r[-1] * crystallizer_volume_square

            D[i] += reduced
            row = np.expand_dims(r, 0)
            B[B_index, :] = volume_average_size_jit(row, volume_fraction_powers, shape_factor, mode=split_ratio)
            B[B_index, -1] = reduced
            B_index += 1

            B[B_index, :] = volume_average_size_jit(row, volume_fraction_powers, shape_factor, mode=1 - split_ratio)
            B[B_index, -1] = reduced
            B_index += 1

    if compression_interval > 0:
        B = compress_jit(B, volume_fraction_powers, shape_factor, compression_interval)

    # restore D to the same rows as original n
    D_original = np.zeros((n_original.shape[0],))
    D_original[included_rows] = D
    return B, D_original
