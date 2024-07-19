import matplotlib.pyplot as plt
import numpy as np

from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range

if __name__ == "__main__":
    length = 20 * 1e-3  # All dimension by default in m
    p1, p2 = np.array([0, 0, -length / 2]), np.array([0, 0, length])
    w = Wire(p1, p2)
    g = Geometry([w])

    fr = np.linspace(2_000, 8_000, 200)  # All dimension by default in MHz
    sc, gain = get_scattering_in_frequency_range(g, fr, eta=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fr, sc)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Forward Scattering (m^2)")
    ax[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax[1].plot(fr, gain)
    ax[1].set_xlabel("Frequency (GHz)")
    ax[1].set_ylabel("S21")

    plt.show()
