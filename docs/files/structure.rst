Project structure
=================


The following graph shows the project directory structure. All files are not displayed as this is a large project.
However, each folder contains a ``README.md`` file to explain its content.

.. code-block:: text

    ├── perfusion/
    │   ├── containers/
    │   │   ├── container.def
    │   │   └── container.sif
    │   ├── boundary_data/
    │   │   ├── templates/
    │   │   └── BC.csv
    │   ├── brain_meshes/
    │   │   └── ...
    │   ├── configs/
    │   │   ├── config_examples/
    │   │   └── ...
    │   ├── hpc_submission/
    │   │   ├── Crescent2/
    │   │   │   └── ...
    │   │   └── Ares/
    │   │   │   └── ...
    │   ├── results/
    │   │   └── ... # All the results files
    │   ├── src/
    │   │   ├── Legacy_version/
    │   │   │   ├── io/
    │   │   │   ├── simulation/
    │   │   │   └── utils/
    │   │   └── X_version/
    │   │   │   ├── io/
    │   │   │   ├── simulation/
    │   │   │   └── utils/
    │   ├── test/
    │   │   └── ...
    │   ├── verification/
    │   │   └── ...
    │   ├── README.md
    │   └── perfusion_runner.sh
    └── docs/
        └── ...

Folder explanation
==================

Perfusion folder
-----------------

This folder contains the code related to the perfusion model, which describe the amount of blood delivered to a given
amount of tissue in a given time (unit describing the perfusion: [ml/min/100g]). This code is structured clearly into
different folders according to the functions of the file.

This folder also contains a ``README.md`` file and a runner, called perfusion_runner.sh, that runs the typical workflow
of the perfusion simulations.

Containers folder
^^^^^^^^^^^^^^^^^

The containers folder groups both ``.def`` and ``.sif`` files related to the containers of the project.
The ``.sif`` file is supposed to be here after the container has been built.

Boundary Data folder
^^^^^^^^^^^^^^^^^^^^

This folder contains the boundary conditions of the brain mesh used for the simulation. A template is available,
or a new file could be generated using the the ``BC_creator.py`` file.

Brain meshes folder
^^^^^^^^^^^^^^^^^^^

This folder contains the description of the brain meshes used for the simulation.

Configs folder
^^^^^^^^^^^^^^

This folder contains the configurations files with the extension ``.yaml``, and a folder gathering some examples of
configurations for the simulations.

HPC submission folder
^^^^^^^^^^^^^^^^^^^^^

This folder contains two sub-folders, Ares and Crescent2, which gather all the submission files related to the HPC
systems supported by the project.

Results
^^^^^^^

This folder gathers all the results from the simulations.

Src folder
^^^^^^^^^^

This folder contains two sub-folders, Legacy_version and X_version, related to the two versions of the FEniCS software.
The project is currently under a modernisation process that consists of switching from FEniCS Legacy to FEniCS-X.
Until now, the X version is not completely implemented and tested, we then decide to conserve the Legacy version in
order to always have a working version of the simulations available.

Both sub-folders are build with 3 folders:

* ``io``: it contains the resources related to the inputs and outputs of the simulations;
* ``simulation``: it contains the scripts of the simulations (i.e. basic_flow_solver.py);
* ``utils``: it is a collection of useful resources.

Test folder
^^^^^^^^^^^

This folder contains the tests of the project, especially the unit tests.

Verification
^^^^^^^^^^^^

This folder contains some initial integration tests.

Docs folder
-----------

This folder contains the source code of the Sphinx documentation of the project.