import matplotlib.pyplot as plt
import numpy as np

from wirenec.geometry import Geometry
from wirenec.geometry.samples import double_SRR
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import plot_geometry


def plate_with_srr():
    R = 25 * 1e-3
    phi = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    x, y = R * np.cos(phi), R * np.sin(phi)

    wires = []
    for i in range(phi.size):
        g = double_SRR(
            inner_radius=2.2 * 1e-3,
            outer_radius=4 * 1e-3,
            wire_radius=0.1 * 1e-3,
            num_of_wires=12,
            delta_phi=np.pi / 2,
            gap=1 * 1e-3,
        )
        g.translate((x[i], y[i], 0))
        wires += g.wires

    return Geometry(wires)


if __name__ == "__main__":
    incidence_params = {"eta": 90, "scattering_phi_angle": 90}
    g = plate_with_srr()

    fr = np.linspace(4_500, 8_500, 100)
    sc, _ = get_scattering_in_frequency_range(g, fr, **incidence_params)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(fr, sc, label="WireNEC")
    ax.plot(*np.load("CST_data.npy"), label="CST")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Forward Scattering (m^2)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.legend()
    plt.show()

    plot_geometry(g, from_top=True)
