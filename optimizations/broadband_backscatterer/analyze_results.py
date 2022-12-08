import numpy as np
import matplotlib.pyplot as plt
import json

from wirenec.visualization import plot_geometry, scattering_plot

from utils import dipolar_limit, get_macros, read_data_cst
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
    i = 3
    s = "iterations_100__N_3__seed_42__frequencies_(7000, 9000, 10000)__scattering_angle_90"
    d, hyperparams = read_data(s)
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

    # x, y = read_data_cst("data/CST/Backward_cst_3x3.txt")
    # ax.plot(x, y, color='b', linestyle='--', label=f'Backward CST')
    # x, y = read_data_cst("data/CST/Forward_cst_3x3.txt")
    # ax.plot(x, y, color='orange', linestyle='--', label=f'Forward CST')

    ax.set_title(sub_folders[i])
    ax.set_xlim(5_000, 14_000)
    ax.legend()
    plt.show()

    with open(f"data/optimization/{sub_folders[i]}/macros.txt", 'w+') as f:
        f.write(get_macros(g_optimized))

    plot_geometry(g_optimized, from_top=True)
