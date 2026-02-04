This repository contains an implementation of a single-GPU ASM solver
developed as part of the thesis "A Nonoverlapping Additive Schwarz Method on a GPU".

The repository is organized as follows:
- `src/`: Contains two Python packages:
  - `dd_solvers/`: Implements the ASM solver, including its local and coarse
    components, as well as the Preconditioned Conjugate Gradient method.
  - `experiments/`: Provides utility functions for mesh generation, problem setup,
    and running experiments.
- `experiments/`: Contains Jupyter notebooks and Python scripts reproducing the
  specific experiments described in the thesis.
- `results/`: Stores the results generated from experiments.
- `notebooks/`: Contains additional Jupyter notebooks for exploratory analysis
  and visualization, i.e. producing figures for the thesis.
- `containers/`: Includes source files for building Docker images to ensure
  a consistent execution environment.
