import dolfinx
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
from matplotlib.collections import LineCollection

__all__ = [
    "Mesh",
    "Mesh2D",
    "Mesh3D",
    "MeshFamily",
]


class Mesh:
    dolfinx_mesh: dolfinx.mesh.Mesh

    def __init__(self, dolfinx_mesh: dolfinx.mesh.Mesh):
        self.dolfinx_mesh = dolfinx_mesh
        self.dolfinx_mesh.topology.create_connectivity(0, self.tdim)
        self.dolfinx_mesh.topology.create_connectivity(self.tdim, self.tdim - 1)

    @property
    def tdim(self):
        return self.dolfinx_mesh.topology.dim

    @property
    def num_cells(self):
        return self.dolfinx_mesh.topology.index_map(self.tdim).size_global

    @property
    def h(self):
        all_cells = np.arange(self.num_cells, dtype=np.int32)
        return self.dolfinx_mesh.h(self.tdim, all_cells)

    @property
    def vertices(self):
        num_local = self.dolfinx_mesh.topology.index_map(0).size_local
        num_ghosts = self.dolfinx_mesh.topology.index_map(0).num_ghosts
        perm = dolfinx.mesh.entities_to_geometry(
            self.dolfinx_mesh,
            0,
            np.arange(
                num_local + num_ghosts,
                dtype=np.int32,
            ),
            False,
        ).flatten()
        return self.dolfinx_mesh.geometry.x[perm, : self.tdim]

    @property
    def elements(self):
        connectivity = self.dolfinx_mesh.topology.connectivity(self.tdim, 0)
        links_per_node = connectivity.array.shape[0] // connectivity.num_nodes
        return connectivity.array.reshape(-1, links_per_node)[: self.num_cells]

    def refine(self, steps=1):
        local_to_global = np.arange(self.num_cells, dtype=np.int32)
        new_mesh = self
        for _ in range(steps):
            new_dolfinx_mesh, parent_cell, _ = dolfinx.mesh.refine(
                new_mesh.dolfinx_mesh, option=dolfinx.mesh.RefinementOption.parent_cell
            )
            new_mesh = self.__class__(new_dolfinx_mesh)
            local_to_global = local_to_global[
                parent_cell[new_mesh.dolfinx_mesh.topology.original_cell_index]
            ]
        return new_mesh, local_to_global


class Mesh2D(Mesh):
    def __init__(self, dolfinx_mesh: dolfinx.mesh.Mesh):
        super().__init__(dolfinx_mesh)

    def plot(self, ax, **kwargs):
        segments = [
            tuple(sorted(segment))
            for element in self.elements
            for segment in self._element_to_segments(element)
        ]
        segments = list(set(segments))
        ax.add_collection(LineCollection(self.vertices[segments], **kwargs))

    def unit_square_uniform(nx: int, ny: int):
        return Mesh2D(
            dolfinx.mesh.create_unit_square(
                comm=MPI.COMM_WORLD,
                nx=nx,
                ny=ny,
                cell_type=dolfinx.mesh.CellType.triangle,
            )
        )

    @staticmethod
    def _element_to_segments(el):
        if len(el) == 3:
            return [
                (el[0], el[1]),
                (el[1], el[2]),
                (el[2], el[0]),
            ]
        elif len(el) == 4:
            return [
                (el[0], el[1]),
                (el[1], el[3]),
                (el[3], el[2]),
                (el[2], el[0]),
            ]
        else:
            raise ValueError(f"Unsupported element with {len(el)} vertices")


class Mesh3D(Mesh):
    def __init__(self, dolfinx_mesh):
        super().__init__(dolfinx_mesh)
        self.dolfinx_mesh.topology.create_entities(1)

    def unit_cube_uniform(nx: int, ny: int, nz: int):
        return Mesh3D(
            dolfinx.mesh.create_unit_cube(
                comm=MPI.COMM_WORLD,
                nx=nx,
                ny=ny,
                nz=nz,
                cell_type=dolfinx.mesh.CellType.tetrahedron,
            )
        )


class MeshFamily:
    def __init__(self, meshes: list[Mesh], mappings: list[np.ndarray]):
        self._meshes = meshes
        self._mappings = mappings

    def __getitem__(self, idx: int) -> Mesh:
        return self._meshes[idx]

    def __len__(self) -> int:
        return len(self._meshes)

    def get_mapping(self, a: int, b: int) -> np.ndarray:
        res = np.arange(self._meshes[a].num_cells, dtype=np.int32)
        for i in range(a - 1, b - 1, -1):
            res = self._mappings[i][res]
        return res

    def refinements(base: Mesh, n: int):
        meshes = [None] * n
        mappings = [None] * (n - 1)
        meshes[0] = base
        for i in range(1, n):
            meshes[i], mappings[i - 1] = meshes[i - 1].refine(1)
        return MeshFamily(meshes=meshes, mappings=mappings)

    def _flat_coords(mesh: Mesh, m: int):
        coords = np.floor(mesh.vertices[mesh.elements].mean(axis=1) * m).astype(int)
        if mesh.tdim == 2:
            return coords[:, 0] + coords[:, 1] * m
        elif mesh.tdim == 3:
            return coords[:, 0] + coords[:, 1] * m + coords[:, 2] * m * m
        else:
            raise ValueError(f"Unsupported dimension {mesh.tdim}")

    def _mapping_to_quadrilateral_uniform(m: int, from_mesh: Mesh, to_mesh: Mesh):
        to_flat_coords = MeshFamily._flat_coords(to_mesh, m)
        coords_to_mesh = to_flat_coords.argsort()

        from_flat_coords = MeshFamily._flat_coords(from_mesh, m)
        return coords_to_mesh[from_flat_coords]

    def uniform_quadrilateral(d: int, n: int):
        meshes = [None] * (n + 1)
        mappings = [None] * n
        m = 2 ** (n - 1)
        if d == 2:
            # Hack to generate mesh with varying diagonal direction
            meshes[-1] = Mesh2D.unit_square_uniform(1, 1).refine(n - 1)[0]
        elif d == 3:
            # Hack to generate mesh with varying diagonal direction
            meshes[-1] = Mesh3D.unit_cube_uniform(1, 1, 1).refine(n - 1)[0]
        else:
            raise ValueError(f"Unsupported dimension {d}")
        for i in range(n - 1, -1, -1):
            if d == 2:
                meshes[i] = Mesh2D(
                    dolfinx.mesh.create_unit_square(
                        comm=MPI.COMM_WORLD,
                        nx=2**i,
                        ny=2**i,
                        cell_type=dolfinx.mesh.CellType.quadrilateral,
                    )
                )
            else:
                meshes[i] = Mesh3D(
                    dolfinx.mesh.create_unit_cube(
                        comm=MPI.COMM_WORLD,
                        nx=2**i,
                        ny=2**i,
                        nz=2**i,
                        cell_type=dolfinx.mesh.CellType.hexahedron,
                    )
                )
            mappings[i] = MeshFamily._mapping_to_quadrilateral_uniform(
                2**i, meshes[i + 1], meshes[i]
            )
        return MeshFamily(meshes=meshes, mappings=mappings)

    def simplices_then_cubes(d: int, fine: int, both: int):
        mesh_family_cubes = MeshFamily.uniform_quadrilateral(d, both + 1)
        mesh_family_simplices = MeshFamily.refinements(
            base=mesh_family_cubes[both + 1],
            n=fine - both + 1,
        )

        meshes = mesh_family_cubes._meshes + mesh_family_simplices._meshes[1:]
        mappings = mesh_family_cubes._mappings + mesh_family_simplices._mappings
        return MeshFamily(meshes=meshes, mappings=mappings)
