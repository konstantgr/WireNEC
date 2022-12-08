import numpy as np
import matplotlib.pyplot as plt

from cmaes import CMA
from tqdm import tqdm
from scipy.stats import linregress

from wirenec.geometry import Geometry
from wirenec.scattering import get_scattering_in_frequency_range

from utils import TrivialObject, mask_geometry


def objective_function(
        params, N=2,
        freq: [list, tuple, np.ndarray] = tuple([10_000]),
        geometry=False, scattering_angle=90
):
    classes_params, lengths_params, angles_params = (params[:N**2].reshape((N, N)),
                                                     params[N**2:2*N**2].reshape((N, N)),
                                                     params[2*N**2:].reshape((N, N)))

    objs = np.empty((N, N), Geometry)
    for i in range(N):
        for j in range(N):
            objs[i, j] = TrivialObject(
                round(classes_params[i, j]), lengths_params[i, j], angles_params[i, j]
            ).object

    g = mask_geometry(objs)
    if not geometry:
        scattering, _ = get_scattering_in_frequency_range(
            g, freq, 90, 90, 90, scattering_angle
        )
        return (-1) * np.prod(scattering)
    else:
        return g


def cma_optimizer(
        iterations=200,
        seed=48,
        frequencies=tuple([9_000]),
        N=2,
        plot_progress=False,
        scattering_angle=90
):
    np.random.seed(seed)

    classes_bounds = [[0, 1] for _ in range(N**2)]
    lengths_bounds = [[0, 1] for _ in range(N**2)]
    angles_bounds = [[0, 2*np.pi] for _ in range(N**2)]

    bounds = np.array(classes_bounds + lengths_bounds + angles_bounds)

    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    mean = lower_bounds + (np.random.rand(3*N**2) * (upper_bounds - lower_bounds))
    sigma = 2 * (upper_bounds[0] - lower_bounds[0]) / 3

    optimizer = CMA(
        mean=mean,
        sigma=sigma,
        bounds=bounds,
        seed=seed,
        population_size=N**2 * 10
    )

    cnt = 0
    max_value, max_params = 0, []

    num_for_progress, slope_for_progress = 100, 1e-8

    pbar = tqdm(range(iterations))
    progress = []

    for generation in pbar:
        solutions = []
        values = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()

            value = objective_function(
                params, freq=frequencies, N=N, scattering_angle=scattering_angle
            )
            values.append(value)
            if abs(value) > max_value:
                max_value = abs(value)
                max_params = params
                cnt += 1

            solutions.append((params, value))

        progress.append(-np.around(np.mean(values), 15))
        if len(progress) > num_for_progress:
            slope1 = linregress(range(len(progress[-num_for_progress:])), progress[-num_for_progress:]).slope
            slope2 = linregress(range(len(progress[-3:])), progress[-3:]).slope

            if abs(slope1) <= slope_for_progress and abs(slope2) <= slope_for_progress:
                print('Minimum slope converged')
                break

        pbar.set_description("Processed %s generation\t max %s mean %s" % (generation, np.around(max_value, 15),
                                                                           -np.around(np.mean(values), 15)))

        optimizer.tell(solutions)

    if plot_progress:
        plt.plot(progress, marker='.', linestyle=':')
        plt.show()

    results = {
        'params': max_params,
        'optimized_value': -max_value,
        'progress': progress,
    }
    return results


