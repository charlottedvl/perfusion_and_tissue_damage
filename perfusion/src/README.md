# Source Code Structure

This folder contains all core logic for simulation, input/output, and utilities. The project supports two variants of the FEniCS framework:

## Legacy_version/

This folder contains code written for the **FEniCS Legacy** stack (based on DOLFIN). 
Use this version if you're running simulations with the classical FEniCS interface.

Submodules:
- `simulation/`: solvers and finite element formulations
- `io/`: mesh and result file handling, format conversion
- `utils/`: reusable helper functions

## X_version/

This folder contains code adapted for **FEniCS-X** (DOLFINx), the modern, high-performance version of FEniCS.

Submodules mirror the same structure:
- `simulation/`
- `io/`
- `utils/`

Use this version if your environment is set up for FEniCS-X.
