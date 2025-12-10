import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# output
output_dir = "../docs/defence-media/"
os.makedirs(output_dir, exist_ok=True)

panel_size = (3.2, 3.2)  # inches
panel_dpi = 200

def fine_diagonals_in_cell(i, j, n):
    """diagonal in fine square cell (i,j) 0-based with alternating orientation"""
    x0, x1 = i / n, (i + 1) / n
    y0, y1 = j / n, (j + 1) / n
    if ((i + j) % 2) == 0:
        return [((x0, y0), (x1, y1))]
    else:
        return [((x0, y1), (x1, y0))]


def diagonal_edges(n):
    segs = []
    for i in range(n):
        for j in range(n):
            segs += fine_diagonals_in_cell(i, j, n)
    return segs

def cube_edges(n: int):
    segs = []
    xs = np.linspace(0, 1, n + 1)
    ys = np.linspace(0, 1, n + 1)
    for i in range(n + 1):
        segs.append(((xs[i], 0.0), (xs[i], 1.0)))
    for j in range(n + 1):
        segs.append(((0.0, ys[j]), (1.0, ys[j])))
    return segs

def set_common_axis(ax):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

def plot_mesh(type: str, m: int):
    n = 2 ** m
    fig, ax = plt.subplots(figsize=panel_size)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="white", zorder=0))

    segs = cube_edges(n)
    if type == "S":
        segs += diagonal_edges(n)
    lc = LineCollection(
        segs, linewidths=1, colors=["black"], zorder=3
    )
    ax.add_collection(lc)
    set_common_axis(ax)

    filename = os.path.join(output_dir, f"mesh_{type}{m}.svg")
    fig.savefig(filename, dpi=panel_dpi, format="svg", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    for n in [0, 1, 2, 3]:
        plot_mesh("C", n)
        plot_mesh("S", n)
