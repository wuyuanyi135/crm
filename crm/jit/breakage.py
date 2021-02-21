import enum
from typing import Optional, Tuple

from numba import jit, njit
import numpy as np

from crm.base.state import State
from crm.jit.compress import compress_jit
from crm.jit.csd import volume_average_size_jit


###############
# Kernels
###############
class BrkKernelType(enum.IntEnum):
    CONSTANT = 0
    CUBIC = 1


@njit(cache=True)
def constant_kernel(l1: float, l2: float):
    return 1


@njit(cache=True)
def cubic_kernel(l1: float, l2: float):
    return (l1 + l2) ** 3


@jit(nopython=True, cache=True)
def binary_breakage_jit(
        n: np.ndarray,
        kernels: np.ndarray,
        volume_fraction_powers, shape_factor,
        compression_interval: float = 0.,
        minimum_count: float = 1000,
        kernel_type: BrkKernelType = BrkKernelType.CONSTANT,
):
    """
    :param kernel_type:
    :param n:
    :param compression_interval:
    :param crystallizer_volume:
    :param minimum_count:
    :param kernels: N x 2 array. N is the combinations. The two columns are split ratio and kernel coefficient. The
    kernel unit should be 1/s. or 1/m^3/s, depending the kernel type.
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
            kernel_coef = kernel[1]

            row = np.expand_dims(r, 0)
            row_0 = volume_average_size_jit(row, volume_fraction_powers, shape_factor, mode=split_ratio)
            row_1 = volume_average_size_jit(row, volume_fraction_powers, shape_factor, mode=1 - split_ratio)
            B[B_index, :] = row_0
            B[B_index + 1, :] = row_1

            if kernel_type == BrkKernelType.CONSTANT:
                val = 1
            elif kernel_type == BrkKernelType.CUBIC:
                val = cubic_kernel(row_0[0], row_1[0])
            else:
                raise ValueError("Unsupported kernel.")

            reduced = kernel_coef * val * r[-1]

            B[B_index, -1] = reduced
            B[B_index + 1, -1] = reduced

            B_index += 2

            D[i] += reduced

    if compression_interval > 0:
        B = compress_jit(B, volume_fraction_powers, shape_factor, compression_interval)

    # restore D to the same rows as original n
    D_original = np.zeros((n_original.shape[0],))
    D_original[included_rows] = D
    return B, D_original


###############
# High level class interface
###############
class BaseBreakage(object):
    def __init__(self, beta_and_ratio: np.ndarray, compression_interval: float = 0., min_count: float = 0.):
        """

        :param beta_and_ratio: The two columns are split ratio and kernel coefficient.
        :param compression_interval:
        :param min_count:
        """
        self.beta_and_ratio = beta_and_ratio
        self.min_count = min_count
        self.compression_interval = compression_interval

    def compute(
            self,
            state: State = None,
            polymorph_idx: int = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """

        :param state:
        :param polymorph_idx:
        :return: B, D
        """
        pass


class ConstantBreakage(BaseBreakage):

    def __init__(self, beta_and_ratio: np.ndarray, compression_interval: float = 0., min_count: float = 0.):
        super().__init__(beta_and_ratio, compression_interval, min_count)

    def compute(
            self,
            state: State = None,
            polymorph_idx: int = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        n = state.n[polymorph_idx]
        form = state.system_spec.forms[polymorph_idx]
        return binary_breakage_jit(n, self.beta_and_ratio, form.volume_fraction_powers, form.shape_factor,
                                   self.compression_interval, kernel_type=BrkKernelType.CONSTANT)


class CubicBreakage(BaseBreakage):

    def __init__(self, beta_and_ratio: np.ndarray, ss_power: float, eps_coef: float,
                 compression_interval: float = 0., min_count: float = 0.):
        super().__init__(beta_and_ratio, compression_interval, min_count)
        self.eps_power = eps_coef
        self.ss_power = ss_power

    def compute(
            self, state: State = None,
            polymorph_idx: int = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        n = state.n[polymorph_idx]
        form = state.system_spec.forms[polymorph_idx]
        eps = state.agitation_power / (state.volume * form.slurry_density(state, polymorph_idx))
        ss = form.state_supersaturation(state, polymorph_idx)

        beta = self.beta_and_ratio.copy()
        beta[:, -1] *= eps ** self.eps_power * ss ** self.ss_power

        return binary_breakage_jit(n, beta, form.volume_fraction_powers, form.shape_factor,
                                   self.compression_interval, kernel_type=BrkKernelType.CUBIC)
