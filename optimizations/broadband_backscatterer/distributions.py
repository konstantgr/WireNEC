import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

from wirenec.geometry import Geometry
from wirenec.visualization import plot_geometry, scattering_plot

from utils import dipolar_limit, get_random_geometry
from optimizer import objective_function


def read_data(folder):
    with open(f"data/optimization/{folder}/optimized_params.json", 'r') as f:
        d = json.load(f)

    with open(f"data/optimization/{folder}/hyperparams.json", 'r') as f:
        hyperparams = json.load(f)

    return d, hyperparams


def get_band(freq, scattering):
    _, y = dipolar_limit(freq)
    limit = y
    idx = np.argwhere(np.diff(np.sign(scattering - limit))).flatten()

    lst = []
    for i in range(idx.size - 1):
        if scattering[idx[i] + 1] - (np.array(y))[idx[i] + 1] > 0:
            lst.append(freq[idx[i + 1]] - freq[idx[i]])

    if len(idx) and scattering[idx[-1] + 1] - (np.array(y))[idx[-1] + 1] > 0:
        lst.append(freq[-1] - freq[idx[-1]])

    band = max(lst) if len(lst) else 0
    return band


def monte_carlo(num=1000):
    bands = []
    for _ in tqdm(range(num)):
        g = get_random_geometry(N=2)

        _, sc = scattering_plot(
            None, g, frequency_start=5_000, eta=90, num_points=100,
            scattering_phi_angle=90,
            label='Optimized Geometry, back scattering'
        )

        band = get_band(np.linspace(5_000, 14_000, 100), sc)
        bands.append(band)
    return np.array(bands)


def analyze_distribution(path):
    data = np.load(path)
    print(
        np.quantile(data, 0.5),
        np.quantile(data, 0.9),
        np.quantile(data, 0.95),
        np.quantile(data, 0.99),
    )
    return data


if __name__ == "__main__":
    base_path = './data/optimization/'
    sub_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    i = 3
    d, hyperparams = read_data(sub_folders[i])
    N = hyperparams['N']

    g_optimized = objective_function(params=np.array(d['params']), N=N, geometry=True)

    # g_optimized = get_random_geometry(N)

    fig, ax = plt.subplots(2, figsize=(6, 8))

    fr, sc = scattering_plot(
        ax[0], g_optimized, frequency_start=5_000, eta=90, num_points=100,
        scattering_phi_angle=90,
        label='Optimized Geometry, back scattering'
    )
    x, y = dipolar_limit(np.linspace(5_000, 14_000, 100))
    ax[0].plot(x, np.array(y), color='b', linestyle='--', label=f'Single Bound')

    ax[0].set_title(sub_folders[i])
    ax[0].legend()

    path = 'data/distributions/band_distribution_2x2.npy'

    calc_data = False
    if calc_data:
        bands = monte_carlo(num=500)
        with open(path, 'wb') as f:
            np.save(f, bands)

    bands = analyze_distribution(path)
    b = get_band(x, sc)

    ax[1].hist(bands, bins=20, density=True)
    ax[1].axvline(b, color='r', linestyle='--')

    plt.show()

    # plot_geometry(g_optimized, from_top=True)
