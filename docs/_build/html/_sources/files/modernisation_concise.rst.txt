Modernisation
=============

Introduction
------------

The GEMINI perfusion model simulates blood flow in brain tissue during acute ischemic stroke using a multi-compartment porous medium framework, implemented in FEniCSx. The pipeline includes:

- **Pre-processing**: Preparing anatomical brain meshes.
- **Permeability Initialization**: Computing tissue permeability tensors.
- **Numerical Solution**: Solving for pressure and velocity fields under healthy or occluded conditions.

The model uses human-readable YAML configuration files to define physical, numerical, and boundary parameters. This guide outlines the requirements and steps to run the model.

Running the Perfusion Model
---------------------------

### Requirements

The model requires the following software (specific versions ensure compatibility):

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
- **Environment**:
  - Python 3.10.x
  - OpenMPI/MPICH 3.x+
  - Linux/Unix

For reproducibility, use the provided Singularity container (see `containers/container.def`).

### Pipeline Steps

1. **Extract Meshes**:
   - Unpack `brain_meshes.tar.xz` into the project directory.

2. **Initialize Permeability**:
   - Run `permeability_initialiser.py` to compute permeability tensors:
     .. code-block:: bash

        mpirun -n 4 python3 src/Gem_X/permeability_initialiser.py

     - Uses `config_permeability_initialiser.yaml`. Takes ~2 minutes with 4 cores.

3. **Solve Flow**:
   - Run `basic_flow_solver.py` for pressure and velocity fields:
     .. code-block:: bash

        mpirun -n 4 python3 src/Gem_X/basic_flow_solver.py

     - Uses `config_basic_flow_solver.yaml`. Takes ~2 minutes with 4 cores.

4. **Handle Occlusions**:
   - Generate boundary conditions for occluded scenarios (e.g., right MCA occlusion):
     .. code-block:: bash

        python3 src/Gem_X/BC_creator.py

     - Update `config_basic_flow_solver.yaml` to point to the generated `BCs_RMCA.csv` and change the output folder to avoid overwriting.

Configuration Files
-------------------

### Permeability Initialiser Config

The `config_permeability_initialiser.yaml` file controls permeability tensor generation:

.. code-block:: yaml

    input:
      mesh_file: '../brain_meshes/b0000/clustered.xdmf'
    output:
      res_fldr: '../brain_meshes/b0000/permeability/'
      save_subres: false
      res_vars: ['K1_form']
    physical:
      e_ref: [0, 0, 1]
      K1_form: [0, 0, 0, 0, 0, 0, 0, 0, 1]

- **input/mesh_file**: Path to the mesh file.
- **output/res_fldr**: Directory for saving tensors.
- **physical/e_ref**: Reference normal vector for cortical surface orientation.
- **physical/K1_form**: Permeability tensor (flattened 3x3 matrix).

### Basic Flow Solver Config

The `config_basic_flow_solver.yaml` file controls the flow simulation:

.. code-block:: yaml

    input:
      healthy: false
      occl_ID: [25]
      read_inlet_boundary: true
      inlet_boundary_file: 'boundary_data/BCs_RMCA.csv'
      mesh_file: '../brain_meshes/b0000/clustered.xdmf'
      permeability_folder: '../brain_meshes/b0000/permeability/'
      inlet_BC_type: 'DBC'
    output:
      res_fldr: '../VP_results/p0000/a/DBC/healthy/read_inlet_true/FE_degree_1/np8/'
      res_vars: ['press1', 'vel1', 'perfusion']
    physical:
      K1gm_ref: 0.001234
      beta12gm: 1.326e-06
      p_arterial: 10000.0
      p_venous: 0.0
      Q_brain: 10.0
    simulation:
      fe_degr: 1
      model_type: 'a'
      vel_order: 1

- **input/healthy**: Toggle healthy (true) or occluded (false) scenarios.
- **input/occl_ID**: List of occluded artery IDs (e.g., 25 for right MCA).
- **output/res_fldr**: Output directory (change to avoid overwriting).
- **physical/K1gm_ref**: Permeability for arterioles.
- **simulation/fe_degr**: Finite element degree (1 for faster runs).

**Tips**:
- Maintain separate config files for different scenarios.
- Ensure `occl_ID` matches mesh or CSV labels.
- Use descriptive `res_fldr` names.
