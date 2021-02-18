from numba import jit
import numpy as np

from crm.jit.compress import compress_jit
from crm.jit.csd import volume_average_size_jit


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
    :param compression_interval: if not zero, compress B with this interval
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
    if nrows == 0:
        return None, None

    combination_index = np.triu_indices(nrows, 0)  # combination can self-intersect!

    # (n choose 2, 2, dim+1) second: two rows for combination
    # combination_table = n[combination_index, :].swapaxes(0, 1)
    # ND indexing is not supported in numba.
    combination_table: np.ndarray = np.empty((combination_index[0].shape[0], 2, ncols))

    for i, (self_row_id, other_row_id) in enumerate(zip(*combination_index)):
        combination_table[i, 0, :] = n[self_row_id, :]
        combination_table[i, 1, :] = n[other_row_id, :]

    D = np.zeros((nrows,))
    # shape: (n choose 2, dim + 1)
    B = np.empty((combination_table.shape[0], ncols))
    for i, row_idx in enumerate(zip(*combination_index)):
        agglomeration_parent_rows = combination_table[i]

        B[i, :] = volume_average_size_jit(agglomeration_parent_rows, volume_fraction_powers, shape_factor, mode=-1.)

        # unit of alpha is #/s. 10.1016/S0009-2509(97)00307-2
        # dN/dt: #/s
        # rate_i == Ni * sum_k(Nk * alpha)

        rate = alpha * np.prod(agglomeration_parent_rows[:, -1]) * crystallizer_volume

        B[i, -1] = rate

        # this for loop should just run twice for binary agglomeration.
        for r in row_idx:
            D[r] += rate

    if compression_interval > 0:
        B = compress_jit(B, volume_fraction_powers, shape_factor, compression_interval)

    # restore D to the same rows as original n
    D_original = np.zeros((n_original.shape[0],))
    D_original[included_rows] = D

    return B, D_original