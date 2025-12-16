from abc import ABC, abstractmethod
import dolfinx
import gmsh
from dolfinx.io import gmsh as gmshio
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
from matplotlib.collections import LineCollection

__all__ = [
    "Mesh",
    "Mesh2D",
    "Mesh3D",
    "MeshFamily",
    "UniformMeshes",
    "UnstructuredMeshes",
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

    def unit_square_unstructured(n: int):
        gmsh.initialize()
        gmsh.model.add("unit_square")

        lc = 1.0 / n  # characteristic length
        # Create corner points
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

        # Create lines
        l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
        l2 = gmsh.model.geo.addLine(p2, p3)  # right
        l3 = gmsh.model.geo.addLine(p3, p4)  # top
        l4 = gmsh.model.geo.addLine(p4, p1)  # left

        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([loop])

        # synchronize geometry
        gmsh.model.geo.synchronize()

        # --- Add physical groups (required by dolfinx.model_to_mesh) ---
        # Physical group for the 2D domain
        phys_domain = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, phys_domain, "Domain")

        # Physical groups for the 1D boundary segments (useful for BCs)
        phys_bottom = gmsh.model.addPhysicalGroup(1, [l1])
        gmsh.model.setPhysicalName(1, phys_bottom, "bottom")
        phys_right = gmsh.model.addPhysicalGroup(1, [l2])
        gmsh.model.setPhysicalName(1, phys_right, "right")
        phys_top = gmsh.model.addPhysicalGroup(1, [l3])
        gmsh.model.setPhysicalName(1, phys_top, "top")
        phys_left = gmsh.model.addPhysicalGroup(1, [l4])
        gmsh.model.setPhysicalName(1, phys_left, "left")

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Convert to dolfinx mesh (gdim=2)
        res = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

        gmsh.finalize()
        return Mesh2D(res.mesh)

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


class MeshFamily(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: str) -> Mesh:
        pass

    @abstractmethod
    def get_mapping(self, from_mesh: str, to_mesh: str) -> np.ndarray:
        pass


class UniformMeshes(MeshFamily):
    def __init__(self, d: int, m: int):
        self.d = d
        self.m = m

        self.s_meshes = [None] * (m + 1)
        self.s_mappings = [None] * (m + 1)
        self.s_meshes[0] = (
            Mesh2D.unit_square_uniform(1, 1)
            if d == 2
            else Mesh3D.unit_cube_uniform(1, 1, 1)
        )
        for i in range(1, m + 1):
            self.s_meshes[i], self.s_mappings[i - 1] = self.s_meshes[i - 1].refine(1)

        self.c_meshes = {}

    def _create_c_mesh(self, level: int):
        if self.d == 2:
            return Mesh2D(
                dolfinx.mesh.create_unit_square(
                    comm=MPI.COMM_WORLD,
                    nx=2**level,
                    ny=2**level,
                    cell_type=dolfinx.mesh.CellType.quadrilateral,
                )
            )
        else:
            return Mesh3D(
                dolfinx.mesh.create_unit_cube(
                    comm=MPI.COMM_WORLD,
                    nx=2**level,
                    ny=2**level,
                    nz=2**level,
                    cell_type=dolfinx.mesh.CellType.hexahedron,
                )
            )

    @property
    def name(self):
        return f"uniform({self.d}D,{self.m})"

    def __getitem__(self, idx: str) -> Mesh:
        if idx.startswith("S"):
            level = int(idx[1:])
            return self.s_meshes[level]
        elif idx.startswith("C"):
            level = int(idx[1:])
            if level not in self.c_meshes:
                self.c_meshes[level] = self._create_c_mesh(level)
            return self.c_meshes[level]
        else:
            raise ValueError(f"Unknown mesh id {idx}")

    def get_mapping(self, from_mesh: str, to_mesh: str) -> np.ndarray:
        from_type, from_level = from_mesh[0], int(from_mesh[1:])
        to_type, to_level = to_mesh[0], int(to_mesh[1:])
        if from_type == "S" and to_type == "S":
            res = np.arange(self.s_meshes[from_level].num_cells, dtype=np.int32)
            for i in range(from_level - 1, to_level - 1, -1):
                res = self.s_mappings[i][res]
            return res
        elif to_type == "C":
            return UniformMeshes._mapping_to_quadrilateral_uniform(
                2**to_level,
                self[from_mesh],
                self[to_mesh],
            )
        else:
            raise ValueError(f"Unsupported mapping from {from_mesh} to {to_mesh}")

    def _flat_coords(mesh: Mesh, m: int):
        coords = np.floor(mesh.vertices[mesh.elements].mean(axis=1) * m).astype(int)
        if mesh.tdim == 2:
            return coords[:, 0] + coords[:, 1] * m
        elif mesh.tdim == 3:
            return coords[:, 0] + coords[:, 1] * m + coords[:, 2] * m * m
        else:
            raise ValueError(f"Unsupported dimension {mesh.tdim}")

    def _mapping_to_quadrilateral_uniform(m: int, from_mesh: Mesh, to_mesh: Mesh):
        to_flat_coords = UniformMeshes._flat_coords(to_mesh, m)
        coords_to_mesh = to_flat_coords.argsort()

        from_flat_coords = UniformMeshes._flat_coords(from_mesh, m)
        return coords_to_mesh[from_flat_coords]


class UnstructuredMeshes(MeshFamily):
    def __init__(self, n: int):
        self.n = n
        self.mesh = Mesh2D.unit_square_unstructured(n)
        self.refinement, self.mapping = self.mesh.refine()

    @property
    def name(self):
        return f"unstructured({self.n})"

    def __getitem__(self, idx):
        if idx == "C":
            return self.mesh
        elif idx == "F":
            return self.refinement
        else:
            raise ValueError(f"Unknown mesh id {idx}")

    def get_mapping(self, from_mesh: str, to_mesh: str) -> np.ndarray:
        if from_mesh == "C" and to_mesh == "C":
            return np.arange(self.mesh.num_cells, dtype=np.int32)
        elif from_mesh == "F" and to_mesh == "C":
            return self.mapping
        elif from_mesh == "F" and to_mesh == "F":
            return np.arange(self.refinement.num_cells, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported mapping from {from_mesh} to {to_mesh}")
