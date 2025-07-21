Requirements
============

The project is currently under a modernisation task to switch from the Legacy version to the X version of the FEniCS
software. Both versions co-exists until the FEniCS-X version is fully functional and tested. We therefore have two
different requirements stack.

For reproducibility, use the provided Singularity container (see ``containers/container.def``). It supports both the
Legacy and X versions of the code.

FEniCS-Legacy
-------------

- **Environment**:

  - Python 3.9.x
  - OpenMPI/MPICH 3.x+
  - Linux/Unix

- **FEniCSx Stack**:

  - fenics (2019.1.0)
  - desist (git+https://github.com/UvaCsl/desist.git#egg=desist)
  - ufl-legacy

- **Linear Algebra/MPI**:

  - petsc4py
  - mpi4py
  - h5py

- **Scientific Python**:

  - numpy (1.21.3)
  - scipy (1.7.1)
  - pandas (1.3.4)
  - PyYAML (6.0.x)

- **Utilities**:

  - tqdm (4.62.3)
  - untangle (1.1.1)
  - matplotlib (3.4.3)
  - docopt (0.6.2)
  - meshio (4.4.6)
  - nibabel (3.2.1)
  - pathos (0.2.8)

FEniCS-X
--------

- **Environment**:

  - Python 3.10.x
  - OpenMPI/MPICH 3.x+
  - Linux/Unix

- **FEniCSx Stack**:

  - fenics-basix (0.9.0)
  - fenics-dolfinx (0.9.0)
  - fenics-ffcx (0.9.0)
  - fenics-ufl (2024.2.0)

- **Linear Algebra/MPI**:

  - petsc4py (3.22.3)
  - mpi4py (4.0.3)
  - h5py (3.13.0)

- **Scientific Python**:

  - numpy (2.2.3)
  - scipy (1.15.2)
  - pandas (2.2.3)
  - PyYAML (6.0.2)

- **Utilities**:

  - joblib (1.4.2)
  - tqdm (4.67.1)
