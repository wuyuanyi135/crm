from numba import jit
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


@jit(nopython=True, cache=True)
def dL_to_dV(
        size: np.ndarray,
        dL: float,
        volume_fraction_powers: np.ndarray,
        shape_factor: float,
        adjust_dim: int = 0
) -> float:
    """
    Convert grid dL into volume coordination dV.
    :param size: row without count column (1D vector)
    :param dL:
    :param volume_fraction_powers:
    :param shape_factor:
    :param adjust_dim: Only support 0
    :return: dV
    """

    # V = prod(size ** vfp) * sf
    # dV/dL1 = prod(size[1:] ** vfp[1:]) * sf * vfp[0] * (size[0]) ** (vfp[0]-1)
    dV = np.prod(size[1:] ** volume_fraction_powers[1:]) * shape_factor * \
         volume_fraction_powers[0] * size[0] ** (volume_fraction_powers[0] - 1) * dL
    return dV


@jit(nopython=True, cache=True, nogil=True)
def volume_average_size_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float, mode: float = 0.):
    """
    Forward to _volume_average_size_jit. Only return the average size
    :param n:
    :param volume_fraction_powers:
    :param shape_factor:
    :param mode:
    :return:
    """
    return _volume_average_size_jit(n, volume_fraction_powers, shape_factor, mode)[0]


@jit(nopython=True, cache=True, nogil=True)
def volume_average_size_and_volume_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float,
                                       mode: float = 0.):
    """
    Forward to _volume_average_size_jit. Get both average size and the volume.
    :param n:
    :param volume_fraction_powers:
    :param shape_factor:
    :param mode:
    :return:
    """
    return _volume_average_size_jit(n, volume_fraction_powers, shape_factor, mode)


@jit(nopython=True, cache=True, nogil=True)
def _volume_average_size_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float, mode: float = 0.):
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
        return np.zeros((n.shape[1],)), 0
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

        return ret, particle_average_volume
    else:
        # multi-dimensional N
        non_first_dim_mean_sizes = (n[:, 1:-1] * np.expand_dims(n[:, -1], 1)).sum(axis=0) / n[:, -1].sum()
        particle_average_volume = particle_average_volume / shape_factor
        non_first_dim_prod = np.prod(non_first_dim_mean_sizes ** volume_fraction_powers[1:])

        # exclude the effect of the non first dimensions. The modified particle_average_volume can be used to
        # calculate the first dimension by reciprocal power of the first dimension
        particle_average_volume = particle_average_volume / non_first_dim_prod

        first_dim_size = particle_average_volume ** (1 / volume_fraction_powers[0])

        ret[0] = first_dim_size
        ret[1:-1] = non_first_dim_prod
        ret[-1] = count

        return ret, particle_average_volume