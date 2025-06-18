Gemini Perfusion Model Modernisation
====================================

.. contents::
   :depth: 3
   :local:

Introduction
------------


The Gemini Perfusion Model aims to simulate cerebral perfusion through porous media by solving Darcy flow equations on patient-specific brain geometries. 
Originally developed using the FEniCS-Legacy finite element library, the model was constrained by challenges including limited parallel scaling, legacy dependency maintenance, and difficulties with extensibility.

The modernisation project addresses these limitations by transitioning the codebase from FEniCS-Legacy to FEniCSx. 
This migration improves the solver's scalability on high-performance computing (HPC) systems, enhances portability through containerisation, and modularises code structure for future extensions such as oxygen transport and thrombosis modelling.

In addition to the core numerical solvers, supporting components such as YAML-based configuration, MPI domain decomposition, and boundary condition generation were redesigned. 
The updated model (referred to as ``Gem_X``) retains compatibility with both single and multi-compartment perfusion (A and ACV models), while achieving significant performance and usability improvements.

This document describes the modernisation methodology, highlights technical changes, evaluates computational scaling, and outlines remaining opportunities for future development.

Background
----------

The original Gemini Perfusion Model was developed using the FEniCS-Legacy finite element library. While it served as a robust tool for simulating cerebral perfusion, several inherent limitations became apparent over time, necessitating a modernization effort.

1. **Technical Debt and Maintenance Challenges**: The Legacy codebase accumulated technical debt due to outdated programming practices and dependencies on obsolete libraries. This made maintenance arduous and increased the risk of introducing bugs during updates. The lack of comprehensive documentation further complicated understanding and modifying the codebase. [1]

2. **Scalability and Performance Constraints**: Designed primarily for serial computations, the Legacy model struggled with scalability. As computational demands grew, especially for high-resolution simulations, the model's performance bottlenecks became more pronounced, limiting its applicability in large-scale studies. [2]

3. **Integration Difficulties**: Integrating the Legacy system with modern tools and platforms was challenging. Its monolithic architecture and reliance on deprecated technologies hindered seamless interoperability with contemporary systems, affecting data exchange and workflow automation. [3]

4. **Security Vulnerabilities**: The use of outdated components exposed the system to potential security threats. Without regular updates and patches, the Legacy model was susceptible to known vulnerabilities, posing risks to data integrity and system reliability. [4]

5. **Resource and Talent Constraints**: As the industry moved forward, finding developers proficient in the technologies used by the Legacy model became increasingly difficult. This scarcity of expertise not only escalated maintenance costs but also impeded the onboarding of new team members. [5]

Recognizing these challenges, the decision to transition to FEniCSx was driven by the need for a more maintainable, scalable, and secure platform. FEniCSx offers enhanced performance, better support for parallel computations, and improved integration capabilities, aligning with the evolving requirements of modern computational modeling.

---

**References**

[1] "Challenges Of Legacy Code In Software Development," Enozom. https://enozom.com/blog/challenges-of-legacy-code-in-software-development/

[2] "Legacy Code: 5 Challenges, Tools and Tips to Overcome Them," Swimm. https://swimm.io/learn/legacy-code/legacy-code-5-challenges-tools-and-tips-to-overcome-them

[3] "Unlocking Value and Ensuring Continuity: Best Practices for Managing Legacy Code," Medium. https://medium.com/tech-lead-hub/unlocking-value-and-ensuring-continuity-best-practices-for-managing-legacy-code-2a06349b267

[4] "Why legacy code is a security risk — and how AI can help," GitLab. https://about.gitlab.com/the-source/security/why-legacy-code-is-a-security-risk-and-how-ai-can-help/

[5] "The Pros and Cons of Working With Legacy Code," Axis Software Dynamics. https://axissoftwaredynamics.com/the-pros-and-cons-of-working-with-legacy-code/

Modernisation Goals
-------------------

The modernisation of the Gemini Perfusion Model was driven by both technical and scientific objectives, reflecting the need to update the software ecosystem to current computational standards while enhancing its scientific capabilities.

Technically, the transition aimed to achieve full MPI-based parallel compatibility, enabling simulations to efficiently utilise high-performance computing (HPC) resources. 
By moving to FEniCSx, the model benefits from better support for domain decomposition and distributed linear algebra solvers, essential for scaling to finer meshes and more complex simulations.

Configuration flexibility was also prioritised. 
The adoption of YAML-based input files allows users to easily modify simulation parameters, boundary conditions, and solver options without needing to alter source code directly. 
This improves reproducibility, facilitates rapid experimentation, and lowers the barrier to entry for new users.

Modularity was a key goal in the redesign. 
The modern codebase separates concerns into distinct, manageable components (e.g., I/O handling, finite element operations, permeability initialisation), improving maintainability and enabling straightforward extension to future applications such as oxygen transport or thrombus dynamics.

Performance improvements were central to the modernisation.
Benchmarks demonstrate that the updated model achieves significantly faster execution times relative to the Legacy code, particularly for larger, parallelised problems.
Despite an initially steeper learning curve due to increased abstraction and a more complex build environment, the modernised model offers greater customisability.
Researchers can more easily integrate new physics, adjust solver settings, or perform novel numerical experiments, making the platform a more powerful and future-proof tool for cerebral perfusion studies.



Overview of Repository Structure
--------------------------------


To support reproducibility and usability, the repository follows the structure outlined below:

.. code-block:: none

├── README.md
├── VP_results
│   └── p0000
├── brain_meshes
│   └── b0000
│       ├── affine_matrices.yaml
│       └──  ...
├── containers
│   ├── container.def
│   └── new_container.def
├── doc
│   ├── Makefile
│   ├── _build
│   │   ├── doctrees
│   │   │   ├── environment.pickle
│   │   │   ├── files
│   │   │   │   ├── Perf_IO.doctree
│   │   │   │   ├── Perf_finite_element_fcts.doctree
│   │   │   │   └── ...
│   │   │   └── index.doctree
│   │   └── html
│   │       ├── .buildinfo
│   │       ├── .buildinfo.bak
│   │       ├── .doctrees
│   │       │   ├── environment.pickle
│   │       │   ├── files
│   │       │   │   ├── Perf_IO.doctree
│   │       │   │   └── ...
│   │       │   └── index.doctree
│   │       ├── _sources
│   │       │   ├── files
│   │       │   │   ├── Perf_IO.rst.txt
│   │       │   │   └── ...
│   │       │   └── index.rst.txt
│   │       ├── _static
│   │       │   ├── alabaster.css
│   │       │   └── ...
│   │       ├── files
│   │       │   ├── Perf_IO.html
│   │       │   └── ...
│   │       └── ...
│   ├── conf.py
│   ├── doctrees
│   │   ├── environment.pickle
│   │   ├── files
│   │   │   ├── Perf_IO.doctree
│   │   │   └── ...
│   │   └── index.doctree
│   ├── fidel_files
│   │   ├── FEniCSx_Oxygen_Model_Test_Form.pdf
│   │   └── ...
│   ├── files
│   │   ├── Perf_IO.rst
│   │   └── ...
│   ├── g2_report
│   │   ├── conclusions.tex
│   │   ├── figures
│   │   │   └── CULogo.png
│   │   ├── front_page.tex
│   │   └── ...
│   ├── html
│   │   ├── .buildinfo
│   │   ├── .buildinfo.bak
│   │   ├── _modules
│   │   │   ├── Perf_IO_fcts.html
│   │   │   └── index.html
│   │   ├── _sources
│   │   │   ├── files
│   │   │   │   ├── Perf_IO.rst.txt
│   │   │   │   └── ...
│   │   │   └── index.rst.txt
│   │   ├── _static
│   │   │   ├── alabaster.css
│   │   │   └── ...
│   │   ├── files
│   │   │   ├── Perf_IO.html
│   │   │   └── ...
│   │   ├── genindex.html
│   │   └── ...
│   ├── index.rst
│   └── make.bat
├── file_path_finder.py
├── perfusion
│   ├── __pycache__
│   │   ├── IO_fcts.cpython-313.pyc
│   │   ├── finite_element_fcts.cpython-313.pyc
│   │   └── suppl_fcts.cpython-313.pyc
│   └── weak_vs_strong
│       └── run1
├── run_this_file.py
├── scripts
│   ├── Ares
│   │   ├── README
│   │   └── ...
│   ├── run_files
│   │   ├── Leg_basic.sh
│   │   └── ...
│   └── sub_scripts
│       └── text.txt
├── src
│   ├── Gem_Legacy
│   │   ├── .dockerignore
│   │   ├── ...
│   │   ├── oxygen
│   │   │   ├── API.py
│   │   │   └── ...
│   │   ├── perfusion
│   │   │   ├── API.py
│   │   │   ├── ...
│   │   │   ├── boundary_data
│   │   │   │   └── BCs.csv
│   │   │   ├── conda_build.def
│   │   │   ├── config_basic_flow_solver.yaml
│   │   │   ├── ...
│   │   │   ├── config_examples
│   │   │   │   ├── config_basic_flow_solver_RMCAo_mod_geom.yaml
│   │   │   │   └── ...
│   │   │   ├── config_permeability_initialiser.yaml
│   │   │   ├── config_templates
│   │   │   │   └── config_base.yaml
│   │   │   ├── convert_msh2hdf5.py
│   │   │   ├── ...
│   │   │   ├── verification
│   │   │   │   ├── Verification_coupled.pvsm
│   │   │   │   ├── ...
│   │   │   │   ├── verification_coupled
│   │   │   │   │   ├── 1-D_Anatomy.txt
│   │   │   │   │   ├── BFOnly.py
│   │   │   │   │   ├── Clots.txt
│   │   │   │   │   ├── bf_sim
│   │   │   │   │   │   └── Model_parameters.txt
│   │   │   │   │   └── config.xml
│   │   │   │   ├── verify_perfusion.sh
│   │   │   │   └── verify_perfusion_coupled.sh
│   │   │   └── weak_vs_strong
│   │   │       ├── plotting
│   │   │       │   ├── speedup_vs_np_a_FE1.png
│   │   │       │   └── ...
│   │   │       ├── run_all_occlusions.sub
│   │   │       └── ...
│   │   ├── perfusion_runner.sh
│   │   ├── perfusion_runner_mod_geom.sh
│   │   ├── requirements.txt
│   │   ├── runner.py
│   │   ├── sensitivity
│   │   │   ├── IO_fcts.py
│   │   │   ├── clear.sh
│   │   │   └── ...
│   │   ├── singularity.def
│   │   ├── test_patient.yml
│   │   └── tissue_health
│   │       ├── API.py
│   │       ├── IO_fcts.py
│   │       ├── README.md
│   │       ├── beta_versions
│   │       │   ├── .gitkeep
│   │       │   └── ...
│   │       ├── config_propagation.yaml
│   │       └── ...
│   └── Gem_X
│       ├── API.py
│       ├── ...
│       ├── configurations
│       │   ├── config_oxygen_solver.yaml
│       │   ├── perfusion
│       │   │   ├── Perfusion_yamls.zip
│       │   │   ├── config_basic_flow_solver.yaml
│       │   │   ├── ...
│       │   │   ├── config_examples
│       │   │   │   ├── config_basic_flow_solver_RMCAo_mod_geom.yaml
│       │   │   │   └── ...
│       │   │   └── config_permeability_initialiser.yaml
│       │   ├── test_patient.yml
│       │   └── tissue_health
│       │       ├── config_propagation.yaml
│       │       ├── config_propagation_LMCAo.yaml
│       │       └── config_tissue_damage.yaml
│       ├── core
│       │   ├── oxygen
│       │   │   ├── API.py
│       │   │   ├── Pytest_Validation
│       │   │   │   ├── README
│       │   │   │   ├── __pycache__
│       │   │   │   │   ├── conftest.cpython-312-pytest-8.3.5.pyc
│       │   │   │   │   └── test_compare_fields.cpython-312-pytest-8.3.5.pyc
│       │   │   │   ├── comparison.def
│       │   │   │   └── ...
│       │   │   ├── README.md
│       │   │   ├── __pycache__
│       │   │   │   ├── FE_solver.cpython-312.pyc
│       │   │   │   └── ...
│       │   │   ├── depth_func_DG.h5
│       │   │   ├── oxygen_main.py
│       │   │   └── to_be_implemented.txt
│       │   ├── perfusion
│       │   │   ├── API.py
│       │   │   ├── ...
│       │   │   ├── Pytest_Validation
│       │   │   │   ├── README
│       │   │   │   ├── __pycache__
│       │   │   │   │   ├── conftest.cpython-312-pytest-8.3.5.pyc
│       │   │   │   │   └── test_compare_fields.cpython-312-pytest-8.3.5.pyc
│       │   │   │   ├── comparison.def
│       │   │   │   └── ...
│       │   │   ├── README.md
│       │   │   ├── __pycache__
│       │   │   │   ├── IO_fcts.cpython-310.pyc
│       │   │   │   └── ...
│       │   │   ├── basic_flow_solver.py
│       │   │   ├── boundary_data
│       │   │   │   └── BCs.csv
│       │   │   ├── coupled_flow_solver.py
│       │   │   ├── ...
│       │   │   └── verification
│       │   │       ├── Verification_coupled.pvsm
│       │   │       ├── ...
│       │   │       ├── verification_coupled
│       │   │       │   ├── 1-D_Anatomy.txt
│       │   │       │   ├── BFOnly.py
│       │   │       │   ├── Clots.txt
│       │   │       │   ├── bf_sim
│       │   │       │   │   └── Model_parameters.txt
│       │   │       │   └── config.xml
│       │   │       ├── verify_perfusion.sh
│       │   │       └── verify_perfusion_coupled.sh
│       │   ├── sensitivity
│       │   │   ├── clear.sh
│       │   │   └── ...
│       │   └── tissue_health
│       │       ├── API.py
│       │       ├── README.md
│       │       ├── beta_versions
│       │       │   ├── .gitkeep
│       │       │   ├── dead_fraction.csv
│       │       │   ├── infarct_estimate.py
│       │       │   └── infarct_estimate_treatment_beta.py
│       │       ├── infarct_estimate_treatment.py
│       │       └── ...
│       ├── functions
│       │   ├── Oxy_FE_solver.py
│       │   ├── ...
│       │   ├── __pycache__
│       │   │   ├── Oxy_FE_solver.cpython-313.pyc
│       │   │   └── ...
│       │   ├── functions_list.txt
│       │   └── ...
│       ├── post_processing
│       │   ├── convert_msh2hdf5.py
│       │   └── ...
│       ├── requirements.txt
│       └── runners
│           ├── build_and_run_docker_image.sh
│           └── ...
└── tests
    └── text.txt

Legend of Key Folders
---------------------

- ``brain_meshes/``: Mesh files, permeability tensors, boundary conditions.  
  - Contains `.msh` mesh files and YAML files for affine matrices and permeability tensors, which define the properties of the simulated brain regions.
  
- ``src/Gem_X/``: Modernised FEniCSx-based implementation.  
  - Code files for the FEniCSx solver and utilities designed for the latest version of the software. 

- ``src/Gem_Legacy/``: Legacy FEniCS-based code.  
  - Older code files based on the legacy FEniCS implementation, for comparison or backward compatibility.

- ``doc/``: Documentation sources and Sphinx build outputs.  
  - Contains `.rst` files for the project's documentation, along with configuration and build files for generating HTML or LaTeX outputs.

- ``containers/``: Apptainer/Singularity container definitions.  
  - Files used for defining container environments for reproducible computing, including YAML configurations.

- ``scripts/``: HPC job submission scripts.  
  - Includes scripts for running simulations and setting up environments in high-performance computing systems.

- ``perfusion/``: Performance testing scripts for strong/weak scaling.  
  - Scripts to analyze the performance of simulations under different load conditions, assessing both strong and weak scaling.

- ``tests/``: Basic unit test placeholders.  
  - Unit tests for various modules and functionality of the codebase, including legacy and modern solvers, utility functions, and models.

Requirements
------------

To ensure reproducibility and compatibility, all simulations were conducted using the following software environment (as shown by ``pip list``). Only the packages directly used in the mesh preprocessing, permeability initialisation, boundary-condition generation, and flow solvers are listed.

Core FEniCSx Stack
------------------

- **fenics-basix** 0.9.0 [Fenics Basix](https://fenicsproject.org)  
- **fenics-dolfinx** 0.9.0 [Fenics DolfinX](https://fenicsproject.org)  
- **fenics-ffcx** 0.9.0 [Fenics FFCx](https://fenicsproject.org)  
- **fenics-ufl** 2024.2.0 [Fenics UFL](https://fenicsproject.org)

Linear Algebra / MPI
----------------------

- **petsc4py** 3.22.3 [petsc4py](https://petsc4py.readthedocs.io)  (PETSc backend for solvers)
- **mpi4py** 4.0.3 [mpi4py](https://mpi4py.readthedocs.io)    (MPI communication)
- **h5py** 3.13.0 [h5py](http://www.h5py.org)    (HDF5 I/O for meshes and results)

Scientific Python Ecosystem
----------------------------

- **numpy** 2.2.3 [numpy](https://numpy.org)    (array manipulations)
- **scipy** 1.15.2 [scipy](https://scipy.org)   (numerical utilities)
- **pandas** 2.2.3 [pandas](https://pandas.pydata.org)   (CSV handling in ``BC_creator.py``)
- **PyYAML** 6.0.2 [PyYAML](https://pyyaml.org)   (YAML configuration parsing)

Utilities / Postprocessing
---------------------------

- **joblib** 1.4.2 [joblib](https://joblib.readthedocs.io)   (optional parallel loops)
- **tqdm** 4.67.1 [tqdm](https://tqdm.github.io)    (progress bars in long loops)

Execution Environment
---------------------

- **Python** 3.10.x [Python](https://www.python.org)
- **MPI** — OpenMPI / MPICH 3.x or higher
- **Linux/Unix** with standard shell tools

Documentation and Compatibility References
-----------------------------------------

For further details on the core components of the FEniCSx stack and their documentation, refer to the following sources:

- **dolfinx**: Python API documentation for ``dolfinx`` (v0.9.0) [Fenics DolfinX](https://fenicsproject.org)
- **FFCx**: Compiler documentation for the FEniCSx FFC compiler (v0.9.0) [Fenics FFCx](https://fenicsproject.org)
- **UFL**: Unified Form Language (UFL) documentation (v2024.2.0) [Fenics UFL](https://fenicsproject.org)
- **Basix**: Documentation for the Basix finite element library (v0.9.0) [Fenics Basix](https://fenicsproject.org)

Containerisation (Recommended)
-----------------------------

For maximal reproducibility, it is recommended to encapsulate this environment in a Docker or Singularity container specifying the above versions. This ensures consistent behavior across different compute nodes and reduces setup overhead on HPC systems. The def files for the containers are given within the containers file with instructions for use.


Perfusion Pipeline
==================

1. Extract the ``brain_meshes.tar.xz`` archive, and place its contents in the main project directory.

2. Compute the permeability tensor by running ``permeability_initialiser.py``. For parallel execution, use:

   .. code-block:: bash

      mpirun -n <number_of_processors> python3 permeability_initialiser.py

   With 4 cores, execution typically completes in under 2 minutes. Parameters are read from the ``config_permeability_initialiser.yaml`` file. This script only needs to be run once, unless permeability configuration parameters are changed—then the tensor must be re-initialised.

3. Solve for the pressure and velocity fields using ``basic_flow_solver.py``. The solver supports parallel execution for healthy perfusion simulations. Example command:

   .. code-block:: bash

      mpirun -n <number_of_processors> python3 basic_flow_solver.py

   Using 4 cores and first-order finite elements, the simulation typically completes in under 2 minutes. Parameters are defined in ``config_basic_flow_solver.yaml``.

4. For occluded scenarios, a boundary condition file must be provided to specify pressure and/or volumetric flow rate on cortical surface territories. An example for right MCA occlusion is included as ``BC_template.csv``. Similar files can be generated by running ``BC_creator.py`` (in serial):

   .. code-block:: bash

      python3 BC_creator.py

   After generating this file, re-run the solver with updated YAML configuration pointing to the new inlet boundary file. Ensure that the ``output/res_fldr`` path is changed to avoid overwriting results from previous simulations.

File Paths
----------

- The ``brain_meshes.tar.xz`` archive should be placed in the main project directory alongside the following subdirectories:
  - ``VP_results/``
  - ``brain_meshes/``
  - ``containers/``
  - ``doc/``
  - ``src/``
  - ``scripts/``
  - ``perfusion/``
  - ``tests/``

- The ``config_permeability_initialiser.yaml`` and ``config_basic_flow_solver.yaml`` files are located in the main directory (or the ``src/`` folder).
- The boundary condition template file ``BC_template.csv`` can be found in the ``perfusion/`` folder, and the results will be saved in the ``output/`` directory.


Boundary Condition Surface Region IDs
-------------------------------------

The ``.csv`` file summarising the boundary conditions uses the following surface region IDs for cortical territories:

- 21 – Left ACA  
- 22 – Left MCA  
- 23 – Left PCA  
- 24 – Right ACA  
- 25 – Right MCA  
- 26 – Right PCA  
- 30 – 
- 4  –   

Configuration Files
===================

Modernized Occlusion Handling via Configuration File
----------------------------------------------------

In the legacy ``FEniCS`` implementation, occlusion scenarios were manually specified through command-line arguments or by modifying the Python source code. For example, a right MCA occlusion would be triggered by executing:

.. code-block:: bash
   :caption: Legacy command-line execution

   python3 BC_creator.py --occluded --occl_ID 25 --mesh_file path/to/mesh.xdmf

This approach relied on ``argparse`` to handle parameter input at runtime. While functional, it suffered from several usability issues:

- Required users to remember and manually supply all flags at each run  
- Increased the likelihood of syntax or logic errors due to missing arguments  
- Coupled execution logic tightly with parameter values, reducing script modularity  
- Made it difficult to reproduce simulations or conduct batch studies across occlusion cases  

In contrast, the updated configuration system in the ``FEniCSx`` implementation encapsulates all relevant inputs in a single YAML configuration file. Occlusion status and arterial selection are defined explicitly using simple flags and lists:

.. code-block:: yaml
   :caption: Occlusion-specific fields in FEniCSx configuration

   input:
     healthy: False
     occl_ID: [25]
     read_inlet_boundary: true
     inlet_boundary_file: boundary_data/BCs_RMCA.csv

This design improves usability and maintainability by separating configuration from code logic. The YAML file is human-readable, version-controllable, and fully reproducible. The benefits include:

- **Explicit case control:** The ``healthy`` flag toggles between healthy and pathological simulations.  
- **Flexible occlusion specification:** Arterial occlusion IDs are provided in the ``occl_ID`` list.  
- **Transparent boundary conditions:** The ``inlet_boundary_file`` points to a CSV file with surface region assignments, pressures, and flow rates.  
- **Improved reproducibility:** Entire simulation setups are defined in one file.  
- **Ease of automation:** Compatible with batch scripts, parameter sweeps, and HPC arrays.  

How to Use the Permeability Initialiser Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The permeability tensor fields are generated by ``permeability_initialiser.py``, which reads the mesh, computes directional tensors, and writes HDF5/XDMF output. All parameters are defined in a YAML file (e.g., ``config_permeability_initialiser.yaml``) with three blocks: ``input``, ``output``, and ``physical``.

1. **input: Mesh file**

   .. code-block:: yaml

      input:
        mesh_file: '../brain_meshes/b0000/clustered.xdmf'

   - **mesh_file**: Path to the labelled tetrahedral mesh with boundary markers.

2. **output: Result storage and resolution**

   .. code-block:: yaml

      output:
        res_fldr: '../brain_meshes/b0000/permeability/'
        save_subres: false
        res_vars: {'K1_form'}

   - **res_fldr**: Directory for saving permeability tensors.  
   - **save_subres**: Save intermediate results (set to true only for debugging).  
   - **res_vars**: Variables to compute (e.g., ``K1_form``).

3. **physical: Tensor definition**

   .. code-block:: yaml

      physical:
        e_ref: [0, 0, 1]
        K1_form: [0, 0, 0, 0, 0, 0, 0, 0, 1]

   - **e_ref**: Reference normal vector for orientation inference.  
   - **K1_form**: Flattened 3×3 tensor defining permeability magnitude and direction.  

Usage Tips:

- Ensure the mesh has correct boundary facets and normals.  
- Adjust ``K1_form`` for anisotropic configurations.  
- Re-run the initialiser if any mesh or physical parameters change.  
- Consistency between mesh and solver is critical.

How to Use the Basic Flow Solver Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pressure and velocity solver is fully driven by ``config_basic_flow_solver.yaml``. This file is split into five blocks: ``input``, ``output``, ``physical``, ``simulation``, and ``optimisation``.

1. **input: Data paths and mode**

   .. code-block:: yaml

      input:
        healthy: False
        occl_ID: [25]
        read_inlet_boundary: true
        inlet_boundary_file: boundary_data/BCs_RMCA.csv
        mesh_file: ../brain_meshes/b0000/clustered.xdmf
        permeability_folder: ../brain_meshes/b0000/permeability/
        inlet_BC_type: DBC

   - **healthy**: Boolean for healthy vs occluded.  
   - **occl_ID**: List of occluded artery IDs.  
   - **read_inlet_boundary**: Read inlet data from CSV.  
   - **inlet_BC_type**: DBC, NBC, or MIXED (if not reading CSV).

2. **output: Results configuration**

   .. code-block:: yaml

      output:
        comp_ave: false
        res_fldr: ../VP_results/p0000/a/DBC/healthy/read_inlet_true/FE_degree_1/np8/
        res_vars: {'press1','press2','press3','vel1','vel2','perfusion','beta12','beta23'}
        integral_vars: {}

   - **res_fldr**: Output directory (must differ between runs).  
   - **res_vars**: Variables to save (pressures, velocities, perfusion, etc.).  
   - **integral_vars**: Optional surface integrals.

3. **physical: Model constants**

   .. code-block:: yaml

      physical:
        K1gm_ref: 0.001234
        beta12gm: 1.326e-06
        p_arterial: 10000.0
        p_venous: 0.0
        Q_brain: 10.0

   - Permeability and coupling coefficients.  
   - Inlet/outlet pressures and total inflow rate.

4. **simulation: Discretisation settings**

   .. code-block:: yaml

      simulation:
        fe_degr: 1
        model_type: 'a'
        vel_order: 1

   - **fe_degr**: Finite element degree for pressure.  
   - **vel_order**: Velocity projection order.  
   - **model_type**: `'a'` (arteriole-only) or `'acv'` (3-compartment).

5. **optimisation: Parameter tuning**

   .. code-block:: yaml

      optimisation:
        parameters: ['gmowm_beta_rat','K1gm_ref']
        random_init: true
        init_param_range: [[0.1,10],[0.0001,0.01]]

   - **parameters**: List to optimise.  
   - **random_init**: Random initialisation flag.  
   - **init_param_range**: Allowed parameter ranges.

Usage Tips:

- Keep separate configs for healthy and occluded runs.  
- Use descriptive paths in ``res_fldr`` to avoid overwrites.  
- Verify ``occl_ID`` matches mesh/CSV labels.  
- Automate swaps with scripts or symbolic links.

Version-Specific Notes
======================

Modernisation Scope
-------------------

- Current main branch focuses on ``Gem_X`` (FEniCSx).
- Oxygen transport and thrombus transport models are in **separate branches**.
- Perfusion models (A, ACV) fully modernised and MPI-compatible.

Known Limitations
-----------------

.. warning::

   Memory allocation issues have been observed when using third-order finite elements (`fe_degr: 3`) in the full three-compartment (`model_type: 'acv'`) model under Dirichlet (DBC), Neumann (NBC), and mixed boundary condition options. The failure occurs during the pressure solve in the `basic_flow_solver.py` stage.

   This limitation prevents running higher-fidelity simulations for the ACV model and hinders performing a proper grid-convergence study alongside the single-compartment (`model_type: 'a'`) solver. Future work should address memory usage and solver scalability for third-order ACV cases to enable these analyses.


Results and Scaling
===================

Weak and Strong Scaling Tests
-----------------------------

.. TODO: Insert tables or plots summarising weak vs strong scaling results (time vs processors, speedup, etc.).

Performance Gains
-----------------

.. TODO: Describe improvements over Legacy (e.g., scalability, parallel efficiency).


Modernisation from FEniCS-Legacy to FEniCSx-0.9
===============================================

The perfusion model comprises three “runner” scripts: ``permeability_initialiser.py``, ``basic_flow_solver.py``, and ``BC_creator.py``. These invoke our supporting libraries (``IO_fcts.py``, ``suppl_fcts.py``, and ``finite_element_fcts.py``). In this section, we illustrate how each runner was modernised to use ``dolfinx-0.9``, providing a direct translation guide.

Permeability Initialiser
------------------------

This script computes an anisotropic permeability tensor field aligned with local vessel orientations. Below we compare key components of the legacy and FEniCSx-0.9 implementations.

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

**Legacy ``dolfin``**:

.. code-block:: python
   :caption: Legacy log suppression

   from dolfin import *
   set_log_level(50)

**Modern ``dolfinx``**:

.. code-block:: python
   :caption: FEniCSx log suppression [dolfinxPyDocs]_

   from dolfinx import log
   log.set_log_level(log.LogLevel.WARNING)

Finite Element Space Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Legacy ``dolfin``**:

.. code-block:: python
   :caption: Legacy tensor function space

   K_space = TensorFunctionSpace(mesh, "DG", 0)

**Modern ``dolfinx``**:

.. code-block:: python
   :caption: FEniCSx tensor-valued space using Basix [basixDocs]_

   import basix.ufl
   element_tensor = basix.ufl.element(
       "DG", "tetrahedron", 0,
       shape=(mesh.geometry.dim, mesh.geometry.dim)
   )
   K_space = fem.functionspace(mesh, element_tensor)

MPI-Aware Output
~~~~~~~~~~~~~~~~

**Legacy ``dolfin``**:

.. code-block:: python
   :caption: Legacy rank-guarded prints

   if rank == 0:
       print("Step 1: Reading input files")

**Modern ``dolfinx``**:

.. code-block:: python
   :caption: FEniCSx print0 helper

   from IO_fcts import print0
   print0("Step 1: Reading input files")

I/O and Data Saving
~~~~~~~~~~~~~~~~~~~

**Legacy ``dolfin``**:

.. code-block:: python
   :caption: Legacy XDMF checkpoint writes

   with XDMFFile(path + 'K1_form.xdmf') as file:
       file.write_checkpoint(K1, "K1_form", 0, XDMFFile.Encoding.HDF5, False)

**Modern ``dolfinx``**:

.. code-block:: python
   :caption: FEniCSx write_function with explicit mesh

   from dolfinx.io import XDMFFile
   with XDMFFile(
         mesh.comm, path + 'K1_form.xdmf', "w",
         encoding=XDMFFile.Encoding.HDF5
       ) as xdmf:
       xdmf.write_mesh(mesh)
       xdmf.write_function(K1, 0)

Each of these updates preserves the original algorithmic flow—reading the mesh, computing vessel orientation, assembling the tensor, and writing results—while adopting the explicit, MPI-aware, and Basix-driven APIs of ``dolfinx-0.9``. This pattern is repeated across the other runner scripts, ensuring a uniform transition from legacy to modern code.


Basic Flow Solver
------------------------

The ``basic_flow_solver.py`` script sets up and solves the multi-compartment Darcy flow problem for cerebral perfusion. Key modernisations for ``dolfinx-0.9`` are highlighted below.

Ghost Mode Configuration
~~~~~~~~~~~~~~~~~~~~~

**Legacy ``dolfin``**:

.. code-block:: python
   :caption: Legacy ghost-mode setting

   # ghost mode options: 'none', 'shared_facet', 'shared_vertex'
   parameters['ghost_mode'] = 'none'

**Modern ``dolfinx``**:

.. code-block:: python
   :caption: FEniCSx ghost-mode setting [dolfinxPyDocs]_

   import dolfinx.mesh
   # Disable ghosting entirely
   mesh_obj = dolfinx.mesh.create_mesh(
       comm,
       topology,
       geometry,
       ghost_mode=dolfinx.mesh.GhostMode.NONE
   )

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

**Legacy ``dolfin``**:

.. code-block:: python
   :caption: Legacy log suppression

   from dolfin import *
   set_log_level(50)

**Modern ``dolfinx``**:

.. code-block:: python
   :caption: FEniCSx log suppression [dolfinxPyDocs]_

   from dolfinx.log import set_log_level, LogLevel
   set_log_level(LogLevel.WARNING)

Argument Parsing and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Legacy argument parsing**:

.. code-block:: python
   :caption: Legacy argument parsing

   parser = argparse.ArgumentParser(...)
   parser.add_argument("--config_file", ...)
   config_file = parser.parse_args().config_file
   configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)

**Modern argument parsing**:

.. code-block:: python
   :caption: FEniCSx argument parsing

   parser = argparse.ArgumentParser(...)
   parser.add_argument(
       "--config_file", type=str,
       default="./config_basic_flow_solver.yaml"
   )
   args = parser.parse_args()
   configs = IO_fcts.basic_flow_config_reader_yml(
       args.config_file, parser
   )

Result Output
~~~~~~~~~~~~~

**Legacy result saving**:

.. code-block:: python
   :caption: Legacy result saving

   with XDMFFile(res_fldr + 'K1_form.xdmf') as f:
       f.write_checkpoint(K1, "K1_form", 0, XDMFFile.Encoding.HDF5, False)
   # timing saved by redirecting sys.stdout to a log file
   old = sys.stdout
   sys.stdout = open(res_fldr + "time.log", "w")
   print("Total time:", total)
   sys.stdout = old

**Modern result saving**:

.. code-block:: python
   :caption: FEniCSx result saving [dolfinxPyDocs]_

   from dolfinx.io import XDMFFile
   with XDMFFile(
         mesh.comm,
         res_fldr + "K1_form.xdmf",
         "w",
         encoding=XDMFFile.Encoding.HDF5
       ) as xdmf:
       xdmf.write_mesh(mesh)
       xdmf.write_function(K1, 0)

   # timing logged via standard file I/O
   with open(res_fldr + "time_info.log", "w") as logf:
       print("Total time:", total, file=logf)

Each update retains the original computational pipeline—reading configuration, allocating function spaces, assembling and solving the variational problem, and writing results—while leveraging ``dolfinx-0.9``’s explicit, MPI-aware API.



Changes in Function-Call Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In migrating the ``basic_flow_solver.py`` from legacy ``dolfin`` to ``dolfinx-0.9``, we retained the overall workflow but refactored many function signatures to expose parameters explicitly and improve readability.

Mesh I/O and Configuration Reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Legacy:**

.. code-block:: python
   :caption: Legacy configuration and mesh I/O

   configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
   mesh, subdomains, boundaries = IO_fcts.mesh_reader(
       configs['input']['mesh_file']
   )

**Modern ``dolfinx-0.9``:**

.. code-block:: python
   :caption: Modern configuration and mesh I/O

   configs = IO_fcts.basic_flow_config_reader_yml(
       args.config_file, parser
   )
   mesh, subdomains, boundaries = IO_fcts.mesh_reader(
       configs['input']['mesh_file']
   )

**Note:** The new reader validates additional fields (e.g., ``healthy``, ``occl_ID``) but the call structure remains unchanged.

Function-Space Allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Legacy:**

.. code-block:: python
   :caption: Legacy function-space allocation

   Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
       fe_mod.alloc_fct_spaces(mesh, fe_degree, model_type, vel_order)

**Modern ``dolfinx-0.9``:**

.. code-block:: python
   :caption: Modern function-space allocation

   Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
       fe_mod.alloc_fct_spaces(
           mesh,
           configs['simulation']['fe_degr'],
           model_type=compartmental_model,
           vel_order=velocity_order
       )

**Change:** Keyword arguments (``model_type``, ``vel_order``) clarify intent and improve extensibility.

Solver Setup
^^^^^^^^^^^^

**Legacy:**

.. code-block:: python
   :caption: Legacy solver setup

   LHS, RHS, sigma1, sigma2, sigma3, BCs = \
       fe_mod.set_up_fe_solver2(
           mesh, subdomains, boundaries,
           Vp, v_1, v_2, v_3, p, p1, p2, p3,
           K1, K2, K3, beta12, beta23,
           p_arterial, p_venous,
           configs['input']['read_inlet_boundary'],
           configs['input']['inlet_boundary_file'],
           configs['input']['inlet_BC_type']
       )

**Modern ``dolfinx-0.9``:**

.. code-block:: python
   :caption: Modern solver setup

   LHS, RHS, sigma1, sigma2, sigma3, BCs = \
       fe_mod.set_up_fe_solver2(
           mesh_obj, subdomains, boundaries,
           Vp, v_1, v_2, v_3, p, p1, p2, p3,
           K1, K2, K3, beta12, beta23,
           p_arterial, p_venous,
           configs['input']['read_inlet_boundary'],
           configs['input']['inlet_boundary_file'],
           configs['input']['inlet_BC_type'],
           model_type=compartmental_model
       )

**Change:** The explicit ``model_type`` keyword avoids hidden defaults and documents the solver’s configuration.

Linear Solver Invocation
^^^^^^^^^^^^^^^^^^^^^^^^

**Legacy:**

.. code-block:: python
   :caption: Legacy linear solver invocation

   p = fe_mod.solve_lin_sys(
       Vp, LHS, RHS, BCs,
       lin_solver, precond, rtol, mon_conv, init_sol
   )

**Modern ``dolfinx-0.9``:**

.. code-block:: python
   :caption: Modern linear solver invocation

   p = fe_mod.solve_lin_sys(
       Vp, LHS, RHS, BCs,
       lin_solver, precond,
       rtol, mon_conv, init_sol,
       inlet_BC_type,
       model_type=compartmental_model
   )

**Change:** Passing ``inlet_BC_type`` allows the solver to adjust PETSc options based on boundary-condition type; ``model_type`` continues to guide internal logic.

Overall, function calls now uniformly use keyword arguments for any non-trivial option. This enhances maintainability, self-documentation, and flexibility for batch experiments or future extensions.```

Boundary Condition Generator Modernisation
------------------------------------------

In the legacy ``FEniCS`` implementation, boundary condition assignment was handled via a script called ``BC_creator.py``, which assigned pressures and volumetric flow rates to surface regions of the cerebral mesh. These values were distributed based on surface area and optionally modified to reflect artery occlusions. Execution relied heavily on command-line arguments passed to ``argparse``, requiring the user to explicitly set flags for occlusions and mesh paths:

.. code-block:: bash
   :caption: Legacy boundary condition invocation

   python3 BC_creator.py --occluded --occl_ID 25 --mesh_file path/to/mesh.xdmf

The new ``FEniCSx``-based implementation shifts to a configuration-driven model, removing the need for command-line specification of occlusions. Instead, simulation state and pathology are encoded directly in a YAML config file:

.. code-block:: yaml
   :caption: Occlusion config snippet

   input:
     healthy: False
     occl_ID: [25]
     inlet_BC_type: MIXED

This enables a significantly cleaner and more modular design. Major differences in the modern implementation include:

- **Centralised Configuration:** All logic related to occlusion state, output folder, and mesh location is controlled from a single YAML file, streamlining batch execution and improving reproducibility.  
- **Surface-Area Scaling:** The total volumetric inflow (``Q_brain``) is distributed across boundary regions proportionally to their surface area, consistent with the legacy implementation.  
- **Arterial Mapping Logic:** Artery IDs are mapped via internal dictionaries to groups of cluster labels. This replaces the external CSV-based ``boundary_mapper.csv`` logic and ensures compatibility across different mesh resolutions.  
- **Mixed Boundary Condition Support:** The new script allows specification of ``DBC``, ``NBC``, or ``MIXED`` inlet conditions. For mixed cases, occluded arteries automatically receive Neumann BCs while others use Dirichlet conditions.  
- **MPI-safe Output:** Output is written only by rank 0, ensuring compatibility with parallel runs.

.. code-block:: python
   :caption: Modern artery-mapped boundary matrix row

   [label, Q_i, P_i, artery_ID, flag]

Arterial Mapping Logic
^^^^^^^^^^^^^^^^^^^^^^

In the legacy implementation, the association between surface region labels and their corresponding feeding arteries was handled via an external CSV file called ``boundary_mapper.csv``. This file contained a manually prepared mapping between mesh surface labels and major cerebral arteries. The code parsed this file and constructed a lookup table used during boundary condition assignment:

.. code-block:: python
   :caption: Legacy mapping from CSV file

   boundary_mapper = np.loadtxt(
       'boundary_mapper.csv', skiprows=1, delimiter=','
   )
   boundary_map = np.zeros(len(boundary_values))
   for idx, val in enumerate(boundary_values):
       boundary_map[idx] = int(
           boundary_mapper[np.argwhere(boundary_mapper[:,1] == val)[0], 0]
       )

Modern Approach: Hardcoded Arterial Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``FEniCSx`` implementation removes this external dependency by introducing an internally defined mapping between cluster IDs and cerebral arteries using structured Python dictionaries:

.. code-block:: python
   :caption: Modern arterial group assignment

   artery_groups = {
       24: list(range(20, 32)),    # Right ACA
       25: list(range(32, 52)),    # Right MCA
       22: list(range(52, 80)),    # Left MCA
       21: list(range(80, 100)),   # Left ACA
       26: list(range(89, 100)),   # Right PCA
       23: list(range(100, 112)),  # Left PCA
       4:  list(range(112, 118)),  # cerebular
       30: list(range(118, 130))   # Brainstem or Circle of Willis
   }

.. code-block:: python
   :caption: Boundary label to artery mapping

   for art_id, label_list in artery_groups.items():
       if label in label_list:
           artery = art_id

This redesign offers several advantages:

- **Mesh Version Independence:** Removes the need to maintain external CSV files for each new mesh discretisation.  
- **Clarity and Control:** Makes arterial mapping transparent and easy to update or extend within the code itself.  
- **Robustness:** Prevents issues arising from corrupted or inconsistent mapping files.  
- **Streamlined Automation:** Batch runs across multiple occlusion configurations do not require separate file dependencies.  

Overall, the switch to internally defined artery groups significantly improves the maintainability and portability of the codebase while preserving the anatomical fidelity of the boundary condition application.```



IO_fcts
-------
the following functions are listed in the order in which they appear within the perfusion model run through. some of these functions are shared by the scripts in the perfusion folder. 

perm_init_config_reader_yml
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note: No changes were made to the code itself, however extensive documentation was added in the DolfinX version for usability.


mesh_reader
^^^^^^^^^^^

***reading mesh***

-Dolfin Version:

  .. code-block:: python

     mesh = Mesh()
     with XDMFFile(comm, mesh_file) as myfile:
         myfile.read(mesh)

  The mesh is directly read into the ``mesh`` object, and the ``XDMFFile`` reads the entire mesh structure in one go.

-DolfinX Version:

  .. code-block:: python

     with XDMFFile(comm, mesh_file, "r") as xdmf_file:
         mesh = xdmf_file.read_mesh(name="mesh")

  The mesh is read using the ``read_mesh`` function and specifically named (``name="mesh"``).

-Improvement:  
  The DolfinX version makes the reading process more explicit, especially when dealing with multiple mesh components, and aligns better with the DolfinX API.

Connectivity Creation
----------------------

-Dolfin Version:  
  There is no explicit creation of connectivity between cells and facets.

-DolfinX Version:

  .. code-block:: python

     mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
     mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

  Connectivity between cells and facets is explicitly created using ``create_connectivity``.

-Improvement: 
  The DolfinX version provides explicit connectivity creation, which is useful for higher-dimensional meshes (e.g., 3D) and when certain operations require this connectivity (e.g., boundary conditions, integrals on facets).

Mesh Validation
---------------

-Dolfin Version: 
  There is no explicit validation for the mesh type.

-DolfinX Version:

  .. code-block:: python

     if not isinstance(mesh, dolfinx.mesh.Mesh):
         raise ValueError(f"Failed to load a valid mesh from {mesh_file}. Check the file format.")

  The DolfinX version includes explicit validation to ensure the mesh is a valid ``dolfinx.mesh.Mesh`` object.

-Improvement:  
  This safeguard ensures the mesh is correctly loaded and is of the expected type, reducing the chance of runtime errors.

Subdomains and Boundaries
--------------------------

- Dolfin Version:

  .. code-block:: python

     subdomains = MeshFunction("size_t", mesh, 3)
     with XDMFFile(comm, mesh_file[:-5]+'_physical_region.xdmf') as myfile:
         myfile.read(subdomains)
     boundaries = MeshFunction("size_t", mesh, 2)
     with XDMFFile(comm, mesh_file[:-5]+'_facet_region.xdmf') as myfile:
         myfile.read(boundaries)

  Subdomains and boundaries are read as ``MeshFunction`` objects.

- DolfinX Version:

  .. code-block:: python

     physical_region_file = mesh_file[:-5] + '_physical_region.xdmf'
     with XDMFFile(comm, physical_region_file, "r") as xdmf_file:
         subdomains = xdmf_file.read_meshtags(mesh, name="mesh")

     facet_region_file = mesh_file[:-5] + '_facet_region.xdmf'
     with XDMFFile(comm, facet_region_file, "r") as xdmf_file:
         boundaries = xdmf_file.read_meshtags(mesh, name="mesh")

  Subdomains and boundaries are read using ``read_meshtags``, which is a more robust method in DolfinX.

- **Improvement:**  
  ``read_meshtags`` ensures better handling for mesh tags in DolfinX.

- **New Complexity:**  
  The DolfinX version introduces error handling using try-except blocks:

  .. code-block:: python

     try:
         with XDMFFile(comm, physical_region_file, "r") as xdmf_file:
             subdomains = xdmf_file.read_meshtags(mesh, name="mesh")
     except Exception as e:
         subdomains = None

  This adds complexity but improves robustness by handling failures gracefully.

Communication Barrier
----------------------

- **Dolfin Version:**  
  No explicit barrier is used.

- **DolfinX Version:**

  .. code-block:: python

     comm.Barrier()

  An MPI barrier is added to synchronize all processes in the communicator.

- **Improvement:**  
  The barrier ensures that all processes in parallel are synchronized, which is important in parallel simulations to avoid race conditions or inconsistent results.

print0
^^^^^^



- **Dolfin Version:**  
  The print statements are wrapped in a check for ``rank == 0`` to ensure only rank 0 prints the output.

- **DolfinX Version:**  
  The ``print0`` function is introduced to wrap print statements with the ``rank == 0`` check, improving code readability and reusability by abstracting the conditional check and supporting arbitrary arguments.



basic_flow_config_reader_yml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note: No changes were made to the code itself, however extensive documentation was added in the DolfinX version for usability.


initialise_permeabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the transition from the legacy Dolfin to the modern DolfinX framework, significant modifications were made to the `initialise_permeabilities` function. These changes were necessitated by the removal of convenient utilities such as `read_checkpoint` and `read_function` from the DolfinX API. As a result, a more manual and modular approach was adopted for reading function data from file.

Main Differences:

- **Loss of `read_checkpoint` and `read_function`**: 
  In Dolfin, permeability tensors were easily restored from `.xdmf` files using simple calls to `read_checkpoint`:

  .. code-block:: python

     K = Function(V)
     K = read_checkpoint(xdmf_file, "K", V)

  In DolfinX, these high-level utilities are no longer available and must be replaced by manual HDF5 operations.

- **Introduction of Two New Functions**:

  `find_dataset_key(f, key)`: A helper to recursively locate the correct dataset path inside a possibly nested HDF5 file:

  .. code-block:: python

     def find_dataset_key(f, key):
         if key in f:
             return key
         for group_key in f:
             if isinstance(f[group_key], h5py.Group):
                 result = find_dataset_key(f[group_key], key)
                 if result is not None:
                     return f"{group_key}/{result}"
         return None

  `read_function_from_h5(K, filename, dataset_name)`: A function to load data into a DolfinX `Function` from HDF5:

  .. code-block:: python

     def read_function_from_h5(K, filename, dataset_name):
         with h5py.File(filename, "r", driver="mpio", comm=K.function_space.mesh.comm) as f:
             dataset_path = find_dataset_key(f, dataset_name)
             full_data = f[dataset_path][:]
         dofmap = K.function_space.dofmap
         local_size = len(dofmap.list.array)
         local_data = full_data[:local_size]
         K.x.array[:] = local_data

- **Increased Coding Complexity**:
  The need to manually manage the data layout, MPI scattering, and ownership of local vs. ghost degrees of freedom has added substantial complexity to the codebase. Instead of a single `read_checkpoint` call, permeability recovery now requires explicit data loading and assignment logic.

- **Greater Flexibility and Customisation**:
  Although the new approach is more verbose, it enables much finer control over:

  - The structure and storage format of HDF5 datasets.
  - How and when data is scattered across MPI processes.
  - Debugging at each stage of data input (e.g., raw shape checking, per-rank printing).

- **Debugging Enhancements**:
  The updated `initialise_permeabilities` function includes detailed printouts (through `print0`) to trace the data loading process, providing better insight during both development and simulation runs.

  For example:

  .. code-block:: python

     from mpi4py import MPI

     def print0(*args, **kwargs):
         if MPI.COMM_WORLD.rank == 0:
             print(*args, **kwargs)


The new design reflects a common theme in DolfinX: trading simplicity for greater flexibility and scalability. While users must now implement some low-level file I/O routines themselves, they gain full transparency over how simulation inputs are handled across distributed memory systems. This design choice better aligns with professional-grade large-scale simulations, at the cost of a steeper learning curve for developers migrating from Dolfin.

- **Explicit MPI Parallelism:**  
  DolfinX consistently enforces MPI parallelism across functions, adding explicit barriers (e.g., ``comm.Barrier()``) and rank-specific operations (e.g., ``print0``), ensuring synchronization and safe distributed execution. Legacy Dolfin code assumed serial or minimally parallel contexts, without explicit synchronization.

- **Structured Mesh and Tag Handling:**  
  In DolfinX, meshes, subdomains, and boundaries are loaded using explicit calls like ``read_mesh`` and ``read_meshtags``, providing clearer structure and error handling. Legacy Dolfin used ``MeshFunction`` without validation or fallback in case of file errors.

- **Manual Data Management:**  
  DolfinX replaces utilities like ``read_checkpoint`` and ``read_function`` with manual HDF5 reading and assignment routines (e.g., ``read_function_from_h5``), giving fine-grained control but requiring more complex, modular code. Dolfin's legacy functions abstracted this into simpler but less flexible calls.

- **Error Handling and Robustness:**  
  DolfinX introduces systematic error checking for mesh validity, file reading, and dataset presence (e.g., ``find_dataset_key``), improving robustness during simulations. In contrast, Dolfin legacy versions often assumed successful reads without verification.

- **Flexibility vs Simplicity Trade-off:**  
  While DolfinX is more verbose and demands careful programming (especially for large parallel runs), it offers superior flexibility, scalability, and debugging capabilities compared to the simpler, more compact Dolfin legacy scripts.


suppl_fcts
----------
The following functions are listed in the order in which they appear within the perfusion model run-through. Some of these functions are shared across scripts in the perfusion model.

comp_vessel_orientation
^^^^^^^^^^^^^^^^^^^^^^^

The function comp_vessel_orientation computes vessel orientation based on a permeability field derived from boundary conditions. Both DolfinX and legacy implementations share the same objective but differ in execution, as outlined below:

.. _comp_vessel_orientation:

comp_vessel_orientation
========================

The function ``comp_vessel_orientation`` computes vessel orientation based on a permeability field derived from boundary conditions. Both DolfinX and legacy implementations share the same objective but differ in execution, as outlined below:

Imports and Dependencies
-------------------------

.. code-block:: python
    :caption: Legacy Dolfin Imports

    from dolfin import *
    import numpy as np

.. code-block:: python
    :caption: DolfinX Imports

    import dolfinx
    import numpy as np
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

- **Reason:** DolfinX requires a more modular import approach due to its separation from the monolithic Dolfin interface.
- **Benefit:** More fine-grained control and compatibility with MPI/PETSc workflows.
- **Negative:** Increased verbosity and steeper learning curve for new users.

Function Space Definition
--------------------------

.. code-block:: python
    :caption: Legacy Function Space

    V = FunctionSpace(mesh, "CG", 1)

.. code-block:: python
    :caption: DolfinX Function Space

    V = fem.FunctionSpace(mesh, ("CG", 1))

- **Reason:** DolfinX enforces an explicit tuple-based function space declaration.
- **Benefit:** Allows greater generality for function space definitions, including mixed and vector-valued spaces.
- **Negative:** Less intuitive for users transitioning from legacy syntax.

Boundary Conditions Setup
--------------------------

.. code-block:: python
    :caption: Legacy DirichletBC

    bc = DirichletBC(V, Constant(0.0), boundary_markers, 1)

.. code-block:: python
    :caption: DolfinX DirichletBC

    fdim = mesh.topology.dim - 1
    boundary_facets = mesh.topology.find_entities(fdim, marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)

- **Reason:** Boundary condition assignment now uses low-level DOF maps rather than string expressions.
- **Benefit:** Enables mesh-independent and scalable boundary condition definitions suitable for parallel environments.
- **Negative:** Requires explicit geometry-based functions or labels, increasing code complexity.

Solver Setup and Solution
--------------------------

.. code-block:: python
    :caption: Legacy Solve

    solve(a == L, u, bc)

.. code-block:: python
    :caption: DolfinX Solve

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
    u = problem.solve()

- **Reason:** DolfinX replaces global `solve` with a `LinearProblem` object.
- **Benefit:** Better encapsulation of linear solver settings and increased solver flexibility.
- **Negative:** More verbose setup and the need to separately handle forward scatter of solutions.

Gradient Projection and Normalization
--------------------------------------

.. code-block:: python
    :caption: DolfinX Gradient Projection

    grad_u = ufl.grad(u)
    projected_grad = fem.Function(W)
    fem.project(grad_u, W, projected_grad)
    normed_grad = projected_grad / ufl.sqrt(ufl.inner(projected_grad, projected_grad))

- **Reason:** Manual gradient computation and projection are more explicit in DolfinX.
- **Benefit:** Allows modular projections into various spaces and better control over function norms.
- **Negative:** Requires additional projection logic and normalization steps.

Parallelization and I/O Handling
--------------------------------

.. code-block:: python
    :caption: DolfinX Parallel Output

    with io.XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(function)

- **Reason:** DolfinX enforces explicit MPI rank management and new XDMF I/O patterns.
- **Benefit:** Ensures correct file handling in distributed runs and compatibility with parallel file formats.
- **Negative:** Requires conditional ``rank == 0`` blocks and manual handling of XDMF I/O lifecycle.

Debugging and Output Messages
-----------------------------

.. code-block:: python
    :caption: Rank-aware Output

    from IO_fcts import print0
    print0("Computation complete.")

- **Reason:** Standard `print` replaced by wrapped logging via ``IO_fcts.print0``.
- **Benefit:** Enables rank-aware logging to avoid duplicate outputs in multi-rank runs.
- **Negative:** Adds reliance on custom utility functions and potential I/O bottlenecks on rank 0.


.. _region_label_assembler:

region_label_assembler
========================

The following examples demonstrate the differences between the Dolfin (legacy) and DolfinX implementations of the ``region_label_assembler`` routine, which is used to gather and broadcast unique region tags across MPI ranks. Key updates include syntax modernization, explicit data casting, and MPI semantics using ``MPI.COMM_WORLD``.

Key Differences and Their Implications
--------------------------------------

MeshTag Access Differences
---------------------------

**DolfinX:**

.. code-block:: python
    :caption: Accessing region labels using DolfinX's MeshTags.values

    region_labels = region.values

**Legacy Dolfin:**

.. code-block:: python
    :caption: Accessing region labels using Dolfin's MeshFunction.array()

    region_labels = region.array()

- **Reason:** DolfinX replaces `array()` with `values` to streamline the interface and better align with Pythonic naming conventions.
- **Benefit:** Easier integration with NumPy and enhanced code clarity.
- **Negative:** Requires adjustment when porting legacy code.

Explicit Casting to ``int64`` and Use of Modulo
------------------------------------------------

**DolfinX:**

.. code-block:: python
    :caption: Forcing int64 casting and positive int32 mapping in DolfinX

    region_labels = np.array(region_labels, dtype=np.int64)
    region_labels = (region_labels % (2**31)).astype(np.int64)

**Legacy Dolfin:**

.. code-block:: python
    :caption: Absence of explicit casting in Dolfin (legacy) implementation

    region_labels = region.array()

- **Reason:** Ensures consistent type across all ranks and avoids integer overflow when working with label values.
- **Benefit:** Cross-platform stability and compatibility, especially in large datasets.
- **Negative:** Added complexity for newcomers or legacy code maintainers.

MPI Interface Differences
--------------------------

**DolfinX:**

.. code-block:: python
    :caption: Modernized MPI initialization using MPI.COMM_WORLD in DolfinX

    comm = MPI.COMM_WORLD

**Legacy Dolfin:**

.. code-block:: python
    :caption: Legacy MPI initialization using MPI.comm_world in Dolfin

    comm = MPI.comm_world

- **Reason:** DolfinX uses standardized mpi4py interfaces to improve readability and align with modern conventions.
- **Benefit:** Future-proof and consistent with other MPI-based libraries.
- **Negative:** Slightly changes the function signature and may confuse those migrating.

Scalar Extraction for Label Count
----------------------------------

**DolfinX:**

.. code-block:: python
    :caption: Explicit scalar casting and type annotations in DolfinX

    n_labels = int(n_labels[0])

**Legacy Dolfin:**

.. code-block:: python
    :caption: Implicit typing and return conversion in Dolfin (legacy)

    n_labels = n_labels[0]

- **Reason:** Explicit scalar conversion is emphasized in DolfinX for type clarity and compatibility.
- **Benefit:** Prevents type errors in strict type-checking environments.
- **Negative:** Verbose and may seem redundant to users familiar with implicit conversions.

perm_tens_comp
^^^^^^^^^^^^^^

comp_transf_mat
^^^^^^^^^^^^^^^

scale_coupling_coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^

scale_permeabilities
^^^^^^^^^^^^^^^^^^^^^

compute_my_variables
^^^^^^^^^^^^^^^^^^^^

project_expression
^^^^^^^^^^^^^^^^^^

project_tensor_expression
^^^^^^^^^^^^^^^^^^^^^^^^^

interpolate_to_mesh_order
^^^^^^^^^^^^^^^^^^^^^^^^^

compute_integral_quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

finite_element_fcts
-------------------

alloc_fct_spaces
^^^^^^^^^^^^^^^^

set_up_fe_solver2: ACV Model Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

set_up_fe_solver2: A Model Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

solve_lin_sys
^^^^^^^^^^^^^

Summary of Modernization of the Cerebral Perfusion Solver
=========================================================

Adapting the Solver to FEniCSx
------------------------------

Boundary Condition Management
------------------------------

Parallelization and Performance Enhancements
---------------------------------------------

Enhanced Solver Configuration and Monitoring
---------------------------------------------

Function Space and Solution Management
---------------------------------------


Summary and Future Work
=======================

Summary of Achievements
------------------------

.. TODO: Briefly summarise what modernisation achieved.

Future Work
-----------

.. TODO: Potential future extensions (full oxygen model integration, dynamic thrombosis models, continuous updating for new DolfinX versions).

References
==========

.. bibliography:: g2_references.bib
   :style: unsrt

