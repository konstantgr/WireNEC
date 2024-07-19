import matplotlib.pyplot as plt
import numpy as np
from geometry_specifications import lengths_B, tau_B

from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import plot_geometry


def create_wire_bundle_geometry(lengths, tau):
    m, n = lengths.shape

    wires = []
    x0, y0 = -(m - 1) * tau / 2, -(n - 1) * tau / 2
    for i in range(m):
        for j in range(n):
            x, y = x0 + i * tau, y0 + j * tau
            p1, p2 = np.array([x, y, -lengths[i, j] / 2]), np.array([x, y, lengths[i, j] / 2])
            wires.append(Wire(p1, p2))

    return Geometry(wires)


if __name__ == "__main__":
    g = create_wire_bundle_geometry(lengths_B, tau_B)

    fr = np.linspace(5_000, 7_000, 200)  # All dimension by default in MHz
    sc, _ = get_scattering_in_frequency_range(g, fr, eta=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fr, sc)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Forward Scattering (m^2)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.show()
    plot_geometry(g)
