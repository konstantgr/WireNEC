import numpy as np

from wirenec.geometry import Geometry


def plane_wave(context, frequency, theta=90, phi=90, eta=0, phi0=270):
    """ Excitation of plane wave on current frequency.
        This function calculate scattering when rp_card called.
    """
    context.fr_card(0, 1,  frequency, 0)
    context.ex_card(1, 1, 1, -1, -1,
                    theta, phi,
                    eta,
                    0, 0, 0)

    context.rp_card(calc_mode=0, n_theta=1, n_phi=1,
                    output_format=1, normalization=0, D=0, A=0,
                    theta0=90, delta_theta=0, phi0=phi0, delta_phi=0,
                    radial_distance=0, gain_norm=0)
    context.xq_card(0)  # Execute simulation
    return context


def get_scattering_in_frequency_range(
        geometry: Geometry, frequency_range: range,
        theta=90, phi=90, eta=90, scattering_phi_angle=270
) -> (np.ndarray, np.ndarray):

    c_const = 299792458
    scattering_array, gain_array = [], []
    wires = geometry.wires
    for frequency in frequency_range:
        lmbda = c_const / (frequency * 1e6)

        g = Geometry(wires)
        g.context = plane_wave(
            g.context, frequency,
            theta, phi, eta, scattering_phi_angle
        )

        rp = g.context.get_radiation_pattern(0)
        scattering_db = rp.get_gain()[0][:]

        scattering = (10.0**(scattering_db / 10.0)) * lmbda**2
        scattering_array.append(scattering)
        gain_array.append(scattering_db)

    scattering_array = np.array(scattering_array).T[0]
    gain_array = np.array(gain_array).T[0]

    return scattering_array, gain_array
