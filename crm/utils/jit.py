from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def particle_volume_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    out = np.ones((n.shape[0],))
    for i in range(n.shape[1] - 1):
        out *= n[:, i] ** volume_fraction_powers[i]
    return out * shape_factor * n[:, -1]


@jit(nopython=True, cache=True)
def volume_fraction_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    return particle_volume_jit(n, volume_fraction_powers, shape_factor).sum(axis=0)


@jit(nopython=True, cache=True)
def volume_average_size_jit(n: np.ndarray, volume_fraction_powers: np.ndarray, shape_factor: float):
    nrows = n.shape[0]
    ncols = n.shape[1]
    if nrows == 0:
        return np.zeros((1, n.shape[1]))
    ret = np.empty((ncols, ))

    count = n[:, -1].sum()
    particle_average_volume = volume_fraction_jit(n, volume_fraction_powers, shape_factor) / count / shape_factor

    if volume_fraction_powers.size == 1:
        # one dimensional
        ret[0] = particle_average_volume ** (1 / volume_fraction_powers[0])
        ret[1] = count
        return ret.reshape((1, 2))
    else:
        # multi-dimensional N
        non_first_dim_mean_sizes = n[:, 1:-1].sum(axis=0) / nrows

        non_first_dim_prod = np.prod(non_first_dim_mean_sizes ** volume_fraction_powers[1:])

        # exclude the effect of the non first dimensions. The modified particle_average_volume can be used to
        # calculate the first dimension by reciprocal power of the first dimension
        particle_average_volume = particle_average_volume / non_first_dim_prod

        first_dim_size = particle_average_volume ** (1 / volume_fraction_powers[0])

        ret[0] = first_dim_size
        ret[1:-1] = non_first_dim_prod
        ret[-1] = count

        return ret.reshape((1, -1))
