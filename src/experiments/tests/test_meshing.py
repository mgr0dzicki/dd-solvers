import pytest
import numpy as np

from experiments import *


def test_unit_square_uniform():
    mesh = Mesh2D.unit_square_uniform(3, 3)
    num_cells = 3 * 3 * 2
    num_vertices = (3 + 1) * (3 + 1)

    assert mesh.tdim == 2
    assert mesh.num_cells == num_cells
    assert mesh.h.shape == (num_cells,)
    assert np.allclose(mesh.h, 2 ** (1 / 2) / 3)
    assert mesh.vertices.shape == (num_vertices, 2)
    assert mesh.elements.shape == (num_cells, 3)


def test_unit_cube_uniform():
    mesh = Mesh3D.unit_cube_uniform(3, 3, 3)
    num_cells = 3 * 3 * 3 * 6
    num_vertices = (3 + 1) * (3 + 1) * (3 + 1)

    assert mesh.tdim == 3
    assert mesh.num_cells == num_cells
    assert mesh.h.shape == (num_cells,)
    assert np.allclose(mesh.h, 3 ** (1 / 2) / 3)
    assert mesh.vertices.shape == (num_vertices, 3)
    assert mesh.elements.shape == (num_cells, 4)
