# Configs

This folder contains YAML configuration files used to define simulation parameters.

Each `.yaml` file includes:
- **Mesh settings** (e.g., input mesh file path)
- **Simulation parameters** (e.g., model type, time steps, solver options)
- **Output settings** (e.g., where results are saved)
- **Physical parameters** (e.g., boundary conditions, material properties)

These config files are loaded by the main scripts to control the simulation behavior without modifying source code.

### Example usage:
```bash
python3 -m src.Legacy_version.simulation.basic_flow_solver
