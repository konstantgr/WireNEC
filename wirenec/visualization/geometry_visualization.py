import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D

kind_colors = {
    None: 'b',
    'SRR': 'r'
}


def plot_geometry(g, from_top=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_params = {
        "linewidth": 3,
        "alpha": 0.8,
        "path_effects": [path_effects.SimpleLineShadow(), path_effects.Normal()]
    }

    if from_top:
        ax.view_init(elev=90, azim=270)

    wires = g.wires
    for wire in wires:
        p1, p2 = wire.p1, wire.p2
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        ax.plot([x1, x2], [y1, y2], [z1, z2],
                color=kind_colors[wire.kind],
                **plot_params)

    # if plane:
    #     x = np.linspace(np.min([wire.p1[0] for wire in wires])*1.2, np.max([wire.p1[0] for wire in wires])*1.2, 100)
    #     y = np.linspace(np.min([wire.p1[1] for wire in wires])*1.2, np.max([wire.p1[1] for wire in wires])*1.2, 100)
    #     x, y = np.meshgrid(x, y)
    #     eq = 0 * x + 0 * y + 0
    #     ax.plot_surface(x, y, eq, color='grey', alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.ticklabel_format(style='sci', scilimits=(0, 0))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    plt.show()
