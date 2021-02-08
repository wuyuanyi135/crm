from typing import Union, List

import numpy as np
from scipy.stats import norm


def edges_to_center_grid(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def compress(n, target):
    hist, edge = np.histogramdd(n[:, :-1], target, weights=n[:, -1])
    has_value = hist > 0
    hist = hist[has_value].reshape((-1, 1))

    grid = [edges_to_center_grid(e) for e in edge]
    mesh_grid = np.meshgrid(*grid)
    mesh_grid = np.stack(mesh_grid, axis=-1)
    mesh_grid = mesh_grid[has_value]

    n_new = np.hstack([mesh_grid, hist])
    return n_new


def create_normal_distribution_n(loc: Union[np.ndarray, List[float]], scale: Union[np.ndarray, List[float]],
                                 count_density=1,
                                 grid_count=50, sigma=2,
                                 grid_fcn=np.linspace):
    assert grid_count > 2
    normvals = []
    grids = []
    loc = np.asarray(loc)
    scale = np.asarray(scale)
    diffs = []
    for l, s in zip(loc.tolist(), scale.tolist()):
        grid_low = np.clip(l - s * sigma, 0, np.inf)
        grid_high = l + s * sigma
        grid = grid_fcn(grid_low, grid_high, grid_count)
        grids.append(grid)
        normval = norm.pdf(grid, loc=l, scale=s)
        normvals.append(normval)
        diffs.append(grid[1] - grid[0])

    diff = np.prod(diffs)
    ndcsd = np.prod(np.stack(np.meshgrid(*normvals), axis=-1), axis=-1).reshape((-1, 1)) * count_density * diff
    sparse_grid = np.stack(np.meshgrid(*grids), axis=-1).reshape((-1, loc.size))

    n = np.hstack([sparse_grid, ndcsd])

    return n