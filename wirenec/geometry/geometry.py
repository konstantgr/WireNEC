from PyNEC import nec_context
import numpy as np


class Wire:
    def __init__(
        self,
        point1: [np.ndarray, list, tuple],
        point2: [np.ndarray, list, tuple],
        radius: float = 0.5*1e-3,
        segments: int = 8,
        conductivity: float = None, kind=None
    ):
        if not (isinstance(point1, np.ndarray) and isinstance(point1, np.ndarray)):
            point1, point2 = np.array(point1), np.array(point2)

        self.p1, self.p2 = point1.ravel(), point2.ravel()
        self.radius = radius
        self.segments = segments

        self.conductivity = conductivity
        self.kind = kind

    @property
    def length(self):
        return np.sqrt(np.sum((self.p1 - self.p2)**2))


class Geometry:
    def __init__(self, wires, scale=1):
        self.wires = wires
        self.scale = scale
        self.context = self.make_context(self.scale)

    def set_wire_conductivity(self, context, conductivity, wire_tag=None):
        """ The conductivity is specified in mhos/meter. Currently all segments of a wire are set. If wire_tag is
        None, all wire_tags are set (i.e., a tag of 0 is used). """
        if wire_tag is None:
            wire_tag = 0

        context.ld_card(5, wire_tag, 0, 0, conductivity, 0, 0)
        # return context

    def make_context(self, scale=1):
        context = nec_context()
        geo = context.get_geometry()

        for idx, wire in enumerate(self.wires):
            if wire.length == 0:
                continue

            # try:
            geo.wire(
                idx + 1, wire.segments,
                *wire.p1, *wire.p2, wire.radius,
                1, 1
            )

        geo.scale(scale)
        context.geometry_complete(0)

        # if hasattr(wire, 'conductivity') and wire.conductivity is not None:
        #     context = self.set_wire_conductivity(context, 0, 1)
        return context

    def translate(self, vector):
        new_wires = []
        for wire in self.wires:
            wire.p1 += vector
            wire.p2 += vector
            new_wires.append(wire)

        self.wires = new_wires
        self.context = self.make_context(self.scale)
        return self

    def rotate(self, alpha=0, beta=0, gamma=0):
        Rz = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)]
        ])
        R = Rz.dot(Ry).dot(Rx)

        new_wires = []
        for wire in self.wires:
            vec1 = np.array([wire.p1]).T
            vec2 = np.array([wire.p2]).T
            vec1_r, vec2_r = R.dot(vec1).T[0], R.dot(vec2).T[0]
            wire.p1 = vec1_r.astype(float)
            wire.p2 = vec2_r.astype(float)

            new_wires.append(wire)

        self.wires = new_wires
        self.context = self.make_context(self.scale)
        return self
