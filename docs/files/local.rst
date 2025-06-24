Running on a local machine
==========================

You can easily run the code on a linux machine. If you have a windows machine, you can use WSL to run a linux
environment on a Windows machine. Find more information about WSL here:
`WSL installation tutorial <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

You need to have a conda environment installed on your Linux environment. If you don't possess one, please enter the
following commands to install Miniconda. The required **version of python is 3.9**.

.. code-block:: bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    source ~/miniconda3/bin/activate
    conda create -n fenics-env -c conda-forge fenics python=3.9
    conda activate fenics-env

Once you have the environment, you can install the required requirements using the following command.

.. code-block:: bash
    pip install -r requirements.txt

Once this is done, you can run the code using a similar command or the perfusion runner.

.. code-block:: bash
    mpirun -n 4 python3.9 permeability_initialiser.py --config_file config_permeability_initialiser.yaml
    # or
    ./perfusion_runner.py
