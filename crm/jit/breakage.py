import enum

from numba import jit
import numpy as np

from crm.jit.compress import compress_jit
from crm.jit.csd import volume_average_size_jit

###############
# Kernels
###############
class BrkKernelType(enum.IntEnum):
    CONSTANT = 0
    SMOLUCHOWSKI = 1
    THOMPSON = 2

@jit(nopython=True, cache=True)
def binary_breakage_jit(
        n: np.ndarray,
        kernels: np.ndarray,
        volume_fraction_powers, shape_factor,
        crystallizer_volume: float,
        compression_interval: float = 0.,
        minimum_count: float = 1000
):
    """
    :param n:
    :param compression_interval:
    :param crystallizer_volume:
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
    ncols = n.shape[1]

    if nrows == 0:
        return None, None

    nkernels = kernels.shape[0]

    D = np.zeros((nrows,))

    # shape: (n * kernels * 2, dim + 1)
    # when evenly break, the two rows will be same.
    B = np.empty((nrows * nkernels * 2, ncols))

    B_index = 0
    for i, r in enumerate(n):
        for _, kernel in enumerate(kernels):
            split_ratio = kernel[0]
            k = kernel[1]
            reduced = k * r[-1] * crystallizer_volume

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