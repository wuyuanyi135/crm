from numba import jit, typed, types
import numpy as np


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

    n_non_adjust_dim = n[:, 1:-1]  # TODO: do not support adjust_dim

    for i, (row, non_adjust_row) in enumerate(zip(n, n_non_adjust_dim)):
        sizes = row[:-1]
        count = row[-1]

        # mapping sizes to indices
        idx = (sizes - dims_min) // interval

        # convert 2d index to 1d index
        flat_idx = int((cumprod_dim * idx).sum())

        vol = np.prod(sizes ** volume_fraction_powers) * shape_factor * count
        volume_grid[flat_idx] += vol
        count_grid[flat_idx] += count

        # this will be count mean, so its weight is count.
        sum_non_adjust_dim_grid[flat_idx] += non_adjust_row * count

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
