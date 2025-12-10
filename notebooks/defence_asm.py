#!/usr/bin/env python3
"""
Generates visualisations for two-grid and three-grid ASM where:
 - local problems show fine elements from ONE solver-mesh element (not the coarse element)
 - outputs are saved as SVG
 - also produces one combined figure (single SVG) showing panels in a row

Usage:
 - edit configuration at top (mode, nx, ny, cx, cy, sx, sy)
 - run: python asm_panels_solver_local_svg.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import os
import sys

# -----------------------
# CONFIGURATION
# -----------------------
mode = sys.argv[1]  # 'two' or 'three'

# fine mesh
nx, ny = 12, 12  # fine mesh resolution (squares)

# coarse mesh (coarsest)
cx, cy = 3, 3

# solver mesh (intermediate) — when mode=='two' solver==coarse
sx, sy = 6, 6

# output
output_dir = "../docs/defence-media/"
os.makedirs(output_dir, exist_ok=True)

# figure styling
panel_size = (3.2, 3.2)  # inches
panel_dpi = 200

# colors (RGBA)
coarse_color = (0.75, 0.9, 1.0, 0.6)  # coarse element fill
solver_color = (0.9, 0.8, 1.0, 0.45)  # solver element fill
local_color = (0.7, 1.0, 0.8, 0.55)  # local patch fill
asm_color = (1.0, 0.9, 0.7, 0.35)  # ASM combined fill

# lines
coarse_linewidth = 1.4
solver_linewidth = 1.0
fine_linewidth = 0.7
coarse_linecolor = (0.05, 0.05, 0.07)
# solver_linecolor = (0.35, 0.1, 0.45)
solver_linecolor = coarse_linecolor
fine_linecolor = (0.15, 0.15, 0.18)

# -----------------------
# Sanity checks / derived quantities
# -----------------------
if mode not in ("two", "three"):
    raise ValueError("mode must be 'two' or 'three'")

# If two-grid, treat solver mesh = coarse mesh
if mode == "two":
    sx, sy = cx, cy

assert nx % sx == 0 and ny % sy == 0, "nx must be divisible by sx and ny by sy"
if mode == "three":
    assert (
        sx % cx == 0 and sy % cy == 0
    ), "sx must be divisible by cx and sy by cy in three-grid mode"

# how many fine cells per solver cell
nx_per_sv = nx // sx
ny_per_sv = ny // sy

# how many fine cells per coarse cell (useful)
nx_per_cx = nx // cx
ny_per_cy = ny // cy


# -----------------------
# Geometry helpers
# -----------------------
def fine_diagonals_in_cell(i, j):
    """diagonal in fine square cell (i,j) 0-based with alternating orientation"""
    x0, x1 = i / nx, (i + 1) / nx
    y0, y1 = j / ny, (j + 1) / ny
    if ((i + j) % 2) == 0:
        return [((x0, y0), (x1, y1))]
    else:
        return [((x0, y1), (x1, y0))]


def fine_diagonals_in_range(ix0, ix1, jy0, jy1):
    segs = []
    for i in range(ix0, ix1 + 1):
        for j in range(jy0, jy1 + 1):
            segs += fine_diagonals_in_cell(i, j)
    return segs


def fine_edges_all():
    segs = []
    xs = np.linspace(0, 1, nx + 1)
    ys = np.linspace(0, 1, ny + 1)
    for i in range(nx + 1):
        segs.append(((xs[i], 0.0), (xs[i], 1.0)))
    for j in range(ny + 1):
        segs.append(((0.0, ys[j]), (1.0, ys[j])))
    return segs


# -----------------------
# Draw helpers (matplotlib)
# -----------------------
def set_common_axis(ax):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_coarse_grid(ax, linewidth=None, color=None):
    if linewidth is None:
        linewidth = coarse_linewidth
    if color is None:
        color = coarse_linecolor
    xs = np.linspace(0, 1, cx + 1)
    ys = np.linspace(0, 1, cy + 1)
    for x in xs:
        ax.plot([x, x], [0, 1], linewidth=linewidth, color=color, zorder=5)
    for y in ys:
        ax.plot([0, 1], [y, y], linewidth=linewidth, color=color, zorder=5)


def draw_solver_grid(ax, linewidth=None, color=None):
    if linewidth is None:
        linewidth = solver_linewidth
    if color is None:
        color = solver_linecolor
    xs = np.linspace(0, 1, sx + 1)
    ys = np.linspace(0, 1, sy + 1)
    for x in xs:
        ax.plot([x, x], [0, 1], linewidth=linewidth, color=color, zorder=6)
    for y in ys:
        ax.plot([0, 1], [y, y], linewidth=linewidth, color=color, zorder=6)


def draw_coarse_fill(ax):
    polys = []
    for ic in range(cx):
        for jc in range(cy):
            x0, x1 = ic / cx, (ic + 1) / cx
            y0, y1 = jc / cy, (jc + 1) / cy
            polys.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    polycol = PolyCollection(
        polys, facecolors=[coarse_color], edgecolors=None, zorder=0
    )
    ax.add_collection(polycol)


def draw_solver_fill(ax):
    polys = []
    for isv in range(sx):
        for jsv in range(sy):
            x0, x1 = isv / sx, (isv + 1) / sx
            y0, y1 = jsv / sy, (jsv + 1) / sy
            polys.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    polycol = PolyCollection(
        polys, facecolors=[solver_color], edgecolors=None, zorder=0
    )
    ax.add_collection(polycol)


# -----------------------
# Panel producers (save as SVG)
# -----------------------
def plot_coarse_panel(filename):
    fig, ax = plt.subplots(figsize=panel_size)
    draw_coarse_fill(ax)
    draw_coarse_grid(ax)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=1.0, color="k", zorder=8)
    set_common_axis(ax)
    fig.savefig(filename, dpi=panel_dpi, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_solver_panel(filename):
    fig, ax = plt.subplots(figsize=panel_size)
    draw_solver_fill(ax)
    draw_solver_grid(ax)
    # show coarse on top to see hierarchy
    draw_coarse_grid(ax, linewidth=0.9, color=coarse_linecolor)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=1.0, color="k", zorder=8)
    set_common_axis(ax)
    fig.savefig(filename, dpi=panel_dpi, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_local_panel_solver_element(isv, jsv, filename):
    """
    Show a local problem that is exactly the fine elements inside solver element (isv,jsv).
    Overlays solver grid and coarse grid so hierarchy is visible.
    """
    fig, ax = plt.subplots(figsize=panel_size)
    # fill solver-element patch with local color
    x0, x1 = isv / sx, (isv + 1) / sx
    y0, y1 = jsv / sy, (jsv + 1) / sy
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=local_color, zorder=0))

    # compute fine indices inside this solver element
    ix0 = isv * nx_per_sv
    ix1 = (isv + 1) * nx_per_sv - 1
    jy0 = jsv * ny_per_sv
    jy1 = (jsv + 1) * ny_per_sv - 1

    # fine vertical lines inside solver element
    xs = np.linspace(ix0 / nx, (ix1 + 1) / nx, (ix1 - ix0 + 2))
    for x in xs:
        ax.plot(
            [x, x], [y0, y1], linewidth=fine_linewidth, color=fine_linecolor, zorder=6
        )

    # fine horizontal lines inside solver element
    ys = np.linspace(jy0 / ny, (jy1 + 1) / ny, (jy1 - jy0 + 2))
    for y in ys:
        ax.plot(
            [x0, x1], [y, y], linewidth=fine_linewidth, color=fine_linecolor, zorder=6
        )

    # fine diagonals inside solver element
    diag_segs = fine_diagonals_in_range(ix0, ix1, jy0, jy1)
    if diag_segs:
        lc = LineCollection(
            diag_segs, linewidths=fine_linewidth, colors=[fine_linecolor], zorder=7
        )
        ax.add_collection(lc)

    # overlay solver & coarse grids
    draw_solver_grid(ax)
    draw_coarse_grid(ax, linewidth=0.9, color=coarse_linecolor)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=1.0, color="k", zorder=10)
    set_common_axis(ax)
    fig.savefig(filename, dpi=panel_dpi, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_asm_panel(filename):
    fig, ax = plt.subplots(figsize=panel_size)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=asm_color, zorder=0))

    # full fine grid lines
    segs = fine_edges_all()
    lc = LineCollection(
        segs, linewidths=fine_linewidth, colors=[fine_linecolor], zorder=3
    )
    ax.add_collection(lc)

    # fine diagonals everywhere
    diag_segs = fine_diagonals_in_range(0, nx - 1, 0, ny - 1)
    if diag_segs:
        lc2 = LineCollection(
            diag_segs, linewidths=fine_linewidth, colors=[fine_linecolor], zorder=4
        )
        ax.add_collection(lc2)

    # overlay solver & coarse grids
    # draw_solver_grid(ax)
    # draw_coarse_grid(ax, linewidth=0.9, color=coarse_linecolor)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=1.0, color="k", zorder=10)
    set_common_axis(ax)
    fig.savefig(filename, dpi=panel_dpi, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_meshes(filename):
    fig, ax = plt.subplots(figsize=panel_size)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, zorder=0, color="white"))

    # full fine grid lines
    segs = fine_edges_all()
    lc = LineCollection(segs, linewidths=0.8, colors=["gray"], zorder=3)
    ax.add_collection(lc)

    # fine diagonals everywhere
    diag_segs = fine_diagonals_in_range(0, nx - 1, 0, ny - 1)
    if diag_segs:
        lc2 = LineCollection(diag_segs, linewidths=0.5, colors=["gray"], zorder=4)
        ax.add_collection(lc2)

    # overlay solver & coarse grids
    draw_solver_grid(ax, linewidth=1.3)
    draw_coarse_grid(ax, linewidth=2.6)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=1.0, color="k", zorder=10)
    set_common_axis(ax)
    fig.savefig(filename, dpi=panel_dpi, format="svg", bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Run generation
# -----------------------
if __name__ == "__main__":
    # filenames include mode
    plot_coarse_panel(os.path.join(output_dir, f"coarse_panel.svg"))
    plot_local_panel_solver_element(
        0, sy - 1, os.path.join(output_dir, f"local_panel_1_{mode}.svg")
    )
    plot_local_panel_solver_element(
        1, sy - 1, os.path.join(output_dir, f"local_panel_2_{mode}.svg")
    )
    plot_local_panel_solver_element(
        sx - 1, 0, os.path.join(output_dir, f"local_panel_last_{mode}.svg")
    )
    plot_asm_panel(os.path.join(output_dir, f"asm_panel.svg"))
    plot_meshes(os.path.join(output_dir, f"meshes_{mode}.svg"))

    print("SVGs written to:", output_dir)
