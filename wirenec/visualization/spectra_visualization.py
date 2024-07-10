from wirenec.geometry import Geometry
from wirenec.scattering import get_scattering_in_frequency_range


def scattering_plot(
    ax,
    geometry: Geometry,
    frequency_start: [float, int] = 2000,
    frequency_finish: [float, int] = 14000,
    theta: [float, int] = 90,
    phi: [float, int] = 90,
    eta: [float, int] = 0,
    scattering_phi_angle: [float, int] = 270,
    log: bool = False,
    num_points: int = 100,
    **plot_dict,
):
    num_points = num_points - 1
    step = (frequency_finish - frequency_start) // num_points
    frequency_range = range(frequency_start, frequency_finish + step, step + 1)

    scattering, _ = get_scattering_in_frequency_range(geometry, frequency_range, theta, phi, eta, scattering_phi_angle)

    if ax:
        ax.set_xlabel("Frequency (MHz)", fontsize=16)
        ax.plot(frequency_range, scattering, **plot_dict)
        if log is True:
            ax.set_yscale("log")

    return frequency_range, scattering


if __name__ == "__main__":
    print("example")
