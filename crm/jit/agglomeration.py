from typing import Tuple, Optional

from numba import jit, njit
import numpy as np
import numba as nb
import enum

from crm.base.state import State
from crm.jit.compress import compress_jit
from crm.jit.csd import volume_average_size_jit


###############
# Kernels
###############
class AgglKernelType(enum.IntEnum):
    CONSTANT = 0
    SMOLUCHOWSKI = 1
    THOMPSON = 2


@njit(cache=True)
def constant_kernel(l1, l2):
    return 1


@njit(cache=True)
def smoluchowski_kernel(l1, l2):
    return (l1 + l2) ** 3


@njit(cache=True)
def thompson_kernel(l1, l2):
    (l1 ** 3 - l2 ** 3) ** 2 / (l1 ** 3 + l2 ** 3)


###############
# Agglomeration interface
###############
@njit(cache=True)
def binary_agglomeration_jit(
        n: np.ndarray,
        volume_fraction_powers: np.ndarray,
        shape_factor: float,
        coef: float,
        compression_interval: float = 0.,
        minimum_count: float = 1000.,
        kernel_type: AgglKernelType = AgglKernelType.CONSTANT
):
    """
    Binary agglomeration according to 10.1016/S0009-2509(00)00059-2.
    The count are in #/m3 (number per volume of crystallizer). The unit of the coef should be m3/s for the constant
    kernel or m3/(m3s) if use the cubic kernel. To use the kinetic parameter from total population (the unit of N is
    #), the coef (1/s) should be multiplied by the crystallizer volume.

    :param minimum_count: when one of the involved class has lower than minimum_count particles, agglomeration is ignored.
    :param compression_interval: if not zero, compress B with this interval
    :param n:
    :param coef: constant coefficient in the kernel term.
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

        L1 = agglomeration_parent_rows[0, 0]
        L2 = agglomeration_parent_rows[1, 0]

        if kernel_type == AgglKernelType.CONSTANT:
            kernel_val = constant_kernel(L1, L2)
        elif kernel_type == AgglKernelType.SMOLUCHOWSKI:
            kernel_val = smoluchowski_kernel(L1, L2)
        elif kernel_type == AgglKernelType.THOMPSON:
            kernel_val = thompson_kernel(L1, L2)
        else:
            raise ValueError("Unsupported kernel")

        rate = coef * kernel_val * np.prod(agglomeration_parent_rows[:, -1])

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


###############
# High end class interface
###############
class BaseAgglomeration(object):
    def __init__(self, compression_interval: float = 0., min_count: float = 0.):
        self.min_count = min_count
        self.compression_interval = compression_interval

    def compute(self, state: State = None, polymorph_idx: int = None) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """

        :param state:
        :param polymorph_idx:
        :return: B, D
        """
        pass


class ConstantAgglomeration(BaseAgglomeration):

    def __init__(self, beta: float, compression_interval: float = 0., min_count: float = 0.):
        """

        :param beta: constant agglomeration kernel (m^3/s)
        """
        super().__init__(compression_interval, min_count)
        self.beta = beta

    def compute(self, state: State = None, polymorph_idx: int = None) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        n = state.n[polymorph_idx]
        form = state.system_spec.forms[polymorph_idx]
        return binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, self.beta,
                                        self.compression_interval, kernel_type=AgglKernelType.CONSTANT)


class SmoluchowskiAgglomeration(BaseAgglomeration):
    def __init__(self, beta: float, ss_power: float, eps_coef0: float, eps_coef1: float,
                 compression_interval: float = 0., min_count: float = 0.):
        """
        Reference: 10.1016/S0009-2509(00)00059-2
        beta_aggl = beta * (1 + eps_coef0 * eps ** 0.5 + eps_coef1 * eps) * ss ** ss_power

        :param beta: m^3/(m^3 * s)
        :param ss_power:
        :param eps_coef0:
        :param eps_coef1:
        :param compression_interval:
        :param min_count:
        """
        super().__init__(compression_interval, min_count)
        self.eps_coef1 = eps_coef1
        self.eps_coef0 = eps_coef0
        self.ss_power = ss_power
        self.beta = beta

    def compute(self, state: State = None, polymorph_idx: int = None) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        n = state.n[polymorph_idx]
        form = state.system_spec.forms[polymorph_idx]
        eps = state.agitation_power / (state.volume * form.slurry_density(state, polymorph_idx))

        ss = form.state_supersaturation(state, polymorph_idx)
        coef = self.beta * (1 + self.eps_coef0 * eps ** 0.5 + self.eps_coef1 * eps) * ss ** self.ss_power

        return binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef,
                                        self.compression_interval, kernel_type=AgglKernelType.SMOLUCHOWSKI)
