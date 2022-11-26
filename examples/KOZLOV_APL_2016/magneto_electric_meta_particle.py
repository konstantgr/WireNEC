import numpy as np
import matplotlib.pyplot as plt

from wirenec.geometry import Wire, Geometry
from wirenec.geometry import SRR
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import plot_geometry


def create_meta_particle(rotate=False):
    length = 23 * 1e-3
    d = 32.3 * 1e-3
    w = Geometry([Wire((-length/2, d, 0), (length/2, d, 0), 1*1e-3)])
    srr = SRR(circle_radius=7*1e-3, gap=2*1e-3, wire_radius=0.05*1e-3, num_of_wires=12)

    g = Geometry(srr.wires + w.wires)
    if rotate:
        g.rotate(np.pi, 0, 0)

    return g, srr, w


if __name__ == "__main__":
    incidence_params = {'eta': 90, 'scattering_phi_angle': 90}
    g, srr, wire = create_meta_particle()
    g_rotated, _, _ = create_meta_particle(rotate=True)

    fr = np.linspace(4_000, 8_500, 100)  # All dimension by default in MHz
    sc, _ = get_scattering_in_frequency_range(g, fr, **incidence_params)
    sc_rotated, _ = get_scattering_in_frequency_range(g_rotated, fr, **incidence_params)

    # sc_srr, _ = get_scattering_in_frequency_range(srr, fr, eta=90, **incidence_params)
    # sc_wire, _ = get_scattering_in_frequency_range(wire, fr, eta=90, **incidence_params)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(fr, sc, label=r'$K-$')
    ax.plot(fr, sc_rotated, label=r'$K+$')

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Forward Scattering (m^2)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.show()

    plot_geometry(g_rotated, from_top=True)

