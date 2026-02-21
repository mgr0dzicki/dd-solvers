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


def test_uniform_meshes():
    mesh_family = UniformMeshes(d=2, m=2)

    mp = mesh_family.get_mapping("S2", "C2")
    from_mesh = mesh_family["S2"]
    to_mesh = mesh_family["C2"]
    assert mp.shape == (4 * 4 * 2,)
    for f, c in enumerate(mp):
        mass_center = from_mesh.vertices[from_mesh.elements[f]].mean(axis=0)
        to_el = to_mesh.vertices[to_mesh.elements[c]]
        x1 = to_el[:, 0].min()
        x2 = to_el[:, 0].max()
        y1 = to_el[:, 1].min()
        y2 = to_el[:, 1].max()
        assert x1 < mass_center[0] < x2
        assert y1 < mass_center[1] < y2

    mp = mesh_family.get_mapping("S2", "S1")
    assert mp.shape == (4 * 4 * 2,)
    for i in range(2 * 2 * 2):
        assert (mp == i).sum() == 4

    mp = mesh_family.get_mapping("S1", "S1")
    assert mp.shape == (2 * 2 * 2,)
    assert np.allclose(mp, np.arange(2 * 2 * 2))

    mp = mesh_family.get_mapping("C2", "C2")
    assert mp.shape == (4 * 4,)
    assert np.allclose(mp, np.arange(4 * 4))
