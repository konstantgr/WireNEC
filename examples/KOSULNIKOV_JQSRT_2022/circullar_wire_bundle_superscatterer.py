import matplotlib.pyplot as plt
import numpy as np

from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import plot_geometry


def get_circullar_geometry(R, N, length, delta_phi=0):
    phi = np.linspace(0, 2 * np.pi, N, endpoint=False) + delta_phi
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    wires = [Wire((x[i], y[i], -length / 2), (x[i], y[i], length / 2)) for i in range(phi.size)]

    return Geometry(wires)


def plot_heatmap(R_array, F_array, f0=6750, R0=20 * 1e-3):
    length, N = 20 * 1e-3, 6
    delta_phi = np.pi / N
    data = np.zeros((R_array.size, F_array.size))

    for i, r in enumerate(R_array):
        g = get_circullar_geometry(r, N, length, delta_phi)
        sc, _ = get_scattering_in_frequency_range(g, F_array, eta=0)
        data[i] = sc

    extent = (F_array.min() / f0, F_array.max() / f0, R_array.min() / R0, R_array.max() / R0)
    plt.imshow(
        data,
        vmax=0.07,
        extent=extent,
        cmap="jet",
        interpolation="nearest",
        origin="lower",
        aspect="auto",
    )
    plt.xlabel(r"$f\; / \; f_0$", fontsize=14)
    plt.ylabel(r"$R \; / \; L$", fontsize=14)
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    R, length, N = 15 * 1e-3, 20 * 1e-3, 6
    delta_phi = np.pi / N

    g = get_circullar_geometry(R, N, length, delta_phi)
    fr = np.linspace(2_000, 12_000, 100)
    sc, _ = get_scattering_in_frequency_range(g, fr, eta=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fr, sc)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Forward Scattering (m^2)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.show()

    plot_geometry(g)

    plot_heatmap(np.linspace(2, 30, 100) * 1e-3, fr)
