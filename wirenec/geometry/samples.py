import numpy as np

from wirenec.geometry import Wire, Geometry
from wirenec.visualization import plot_geometry


def SRR(
    circle_radius=7.3*1e-3,
    gap=1.5 * 1e-3,
    delta_phi=-np.pi/2,
    wire_radius=0.5 * 1e-3,
    num_of_wires=12
):
    n, r = num_of_wires, circle_radius
    phi0 = np.arctan(gap/2 / r)  # to create free space at Y
    phi = np.linspace(phi0, np.pi * 2 - phi0, n+1, endpoint=True) + delta_phi

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = 0

    wires = []
    for i in range(n):
        p0, p1 = (x[i], y[i], 0), (x[i + 1], y[i + 1], 0)
        # w = Wire(p0, p1, wire_radius, segments=2, kind='SRR')
        wires.append(Wire(p0, p1, wire_radius, segments=2, kind='SRR'))

    return Geometry(wires)


def double_SRR(
    inner_radius=7.3*1e-3,
    outer_radius=5.3*1e-3,
    delta_phi_inner=np.pi,
    gap=1.5 * 1e-3,
    delta_phi=-np.pi/2,
    wire_radius=0.5 * 1e-3,
    num_of_wires=12
):
    g_inner = SRR(inner_radius, gap, delta_phi + delta_phi_inner, wire_radius, num_of_wires)
    g_outer = SRR(outer_radius, gap, delta_phi, wire_radius, num_of_wires)
    return Geometry(g_inner.wires + g_outer.wires)


def double_srr_6GHz(
        r=3.25 * 1e-3,
        p0=(0, 0, 0),
        wr=0.25*1e-3,
        num=12):
    g = double_SRR(
        inner_radius=r - 1e-3, outer_radius=r,
        wire_radius=wr, num_of_wires=num
    )
    g.translate(p0)

    return g


if __name__ == "__main__":
    g = double_SRR()
    plot_geometry(g, from_top=True)
