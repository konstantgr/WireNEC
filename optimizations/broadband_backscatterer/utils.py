import numpy as np

from wirenec.geometry import Wire, Geometry
from wirenec.geometry.samples import double_srr_6GHz
from wirenec.scattering import get_scattering_in_frequency_range


class TrivialObject:
    def __init__(self, object_type, length_ratio, angle):
        __lengths_wire = (5 * 1e-3, 20 * 1e-3)
        __radiuses_srr = (3.25 * 1e-3, 9 * 1e-3)

        self.size_parameter = (
                __lengths_wire[0] + (__lengths_wire[1] - __lengths_wire[0]) * length_ratio
                if object_type == 0
                else __radiuses_srr[0] + (__radiuses_srr[1] - __radiuses_srr[0]) * length_ratio
        )
        self.object_type = object_type
        self.angle = angle

    @property
    def object(self):
        if self.object_type == 0:
            g = Geometry([Wire((0, -self.size_parameter/2, 0), (0, self.size_parameter/2, 0), 0.5*1e-3)])
        elif self.object_type == 1:
            g = double_srr_6GHz(self.size_parameter)
        else:
            return Exception

        g.rotate(self.angle, 0, 0)
        return g


def dipolar_limit(freq):
    c = 299792458
    lbd = c / (freq * 1e6)
    lengths = lbd / 2

    res = []
    for i, l in enumerate(lengths):
        g = Geometry([Wire((0, 0, -l/2), (0, 0, l/2), 0.5*1e-3)])
        f = freq[i]
        scattering = get_scattering_in_frequency_range(g, [f], 90, 90, 0)
        res.append(scattering[0])

    return freq, res


def mask_geometry(
    objs,
    tau_x=20*1e-3,
    tau_y=20*1e-3,
):
    np.random.seed(42)

    M, N = objs.shape
    a_x = tau_x * N
    a_y = tau_y * M

    x0, y0 = -a_x/2 + tau_x/2, -a_y/2 + tau_y/2

    wires = []
    for i in range(N):
        for j in range(M):
            x, y = x0 + tau_x * i, y0 + tau_y * j
            g = objs[j, i]
            g.translate((x, y, 0))

            wires += g.wires

    return Geometry(wires)
