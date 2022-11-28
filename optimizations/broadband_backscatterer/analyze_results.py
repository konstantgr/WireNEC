import numpy as np
import matplotlib.pyplot as plt
import json

from wirenec.visualization import plot_geometry, scattering_plot

from utils import dipolar_limit, get_macros
from optimizer import objective_function


def read_data(folder):
    with open(f"data/optimization/{folder}/optimized_params.json", 'r') as f:
        d = json.load(f)

    with open(f"data/optimization/{folder}/hyperparams.json", 'r') as f:
        hyperparams = json.load(f)

    return d, hyperparams


if __name__ == "__main__":
    import os

    base_path = './data/optimization/'
    sub_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    i = 5
    d, hyperparams = read_data(sub_folders[i])
    N = hyperparams['N']

    g_optimized = objective_function(params=np.array(d['params']), N=N, geometry=True)

    fig, ax = plt.subplots(1, figsize=(8, 6))

    scattering_plot(
        ax, g_optimized, eta=90, num_points=100,
        scattering_phi_angle=90,
        label='Optimized Geometry, back scattering'
    )

    scattering_plot(
        ax, g_optimized, eta=90, num_points=100,
        scattering_phi_angle=270,
        label='Optimized Geometry, forward scattering'
    )

    x, y = dipolar_limit(np.linspace(5_000, 14_000, 100))
    ax.plot(x, np.array(y)*N**2, color='b', linestyle='--', label=f'{N**2} Bound')

    ax.set_title(sub_folders[i])
    ax.set_xlim(5_000, 14_000)
    ax.legend()
    plt.show()

    with open(f"data/optimization/{sub_folders[i]}/macros.txt", 'w+') as f:
        f.write(get_macros(g_optimized))

    # plot_geometry(g_optimized, from_top=True)
