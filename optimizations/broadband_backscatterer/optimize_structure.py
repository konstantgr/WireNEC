import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from wirenec.geometry import Geometry
from wirenec.visualization import plot_geometry, scattering_plot

from utils import TrivialObject, mask_geometry, dipolar_limit, get_random_geometry
from optimizer import cma_optimizer, objective_function


if __name__ == "__main__":
    N = 2
    seed = 42

    hyperparams = {
        'iterations': 100, 'N': N, 'seed': seed,
        "frequencies": (9_000, 10_000, 10_500),
    }
    d = cma_optimizer(**hyperparams)
    g_random = get_random_geometry(N=N)
    g_optimized = objective_function(params=d['params'], N=N, geometry=True)

    path = "data/optimization/"
    for param, value in hyperparams.items():
        path += f"{param}_{str(value)}__"
    path = path.rstrip("_")

    Path(path).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, figsize=(6, 8))

    scattering_plot(
        ax[0], g_random, eta=90, num_points=100,
        scattering_phi_angle=90,
        label='Non-optimized Geometry'
    )

    scattering_plot(
        ax[0], g_optimized, eta=90, num_points=100,
        scattering_phi_angle=90,
        label='Optimized Geometry'
    )

    x, y = dipolar_limit(np.linspace(5_000, 14_000, 100))

    ax[0].plot(x, np.array(y)*N**2, color='b', linestyle='--', label=f'{N**2} Bound')
    ax[0].plot(x, np.array(y), color='k', linestyle='--', label=f'Single dipole bound')
    ax[1].plot(d['progress'], marker='.', linestyle=':')

    ax[0].set_xlim(5_000, 14_000)
    ax[0].legend()

    fig.savefig(f'{path}/scattering_progress.pdf', dpi=200, bbox_inches='tight')
    plt.show()

    plot_geometry(g_optimized, from_top=True, save_to=f'{path}/optimized_geometry.pdf')

    with open(f'{path}/hyperparams.json', 'w+') as fp:
        json.dump(hyperparams, fp)
    with open(f'{path}/optimized_params.json', 'w+') as fp:
        d['params'] = d['params'].tolist()
        json.dump(d, fp)
    with open(f'{path}/progress.npy', 'wb') as fp:
        np.save(fp, np.array(d['progress']))
