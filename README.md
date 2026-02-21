# Massively Parallel Domain Decomposition Preconditioner on a GPU: Efficient Implementation and Fine-Tuning

This repository contains a single-GPU implementation of the Additive
and Hybrid Schwarz methods for elliptic PDEs.

## Installation

The core solver is contained in the `dd_solvers` package located
in `src/`. Since the package builds PyTorch C&#8288;+&#8288;+ extensions, it must be
installed without build isolation:
```bash
pip install --no-build-isolation src/dd_solvers
```
To ensure all dependencies are properly configured, we recommend using the Docker
container defined in the `containers/` directory and available on DockerHub as:
```
mgrodzicki/dd-solvers:pytorch-24.12
```

## Benchmarks and Experiments

Utilities for benchmarking the solver on the model problem are provided
in the `src/experiments` package. Scripts for reproducing specific benchmarks
are located in `experiments/`, while the obtained results are stored in `results/`.
