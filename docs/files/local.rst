Running on a local machine
==========================

Cloning the GitHub repository
-----------------------------

Before running or editing the code, you need to clone it. You can use the following command:

.. code-block:: bash

    git clone https://github.com/Gemini-DTH/perfusion_and_tissue_damage.git


Operating system recommended
----------------------------

You can easily run the code on a Linux machine. If you have a windows machine, you can use WSL to run a linux
environment on a Windows machine. Find more information about WSL here:
`WSL installation tutorial <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

Installing a package/environment manager
----------------------------------------

You need to have a conda environment installed on your Linux environment. If you don't possess one, please enter the
following commands to install Miniconda. The required **version of python is 3.9**.

.. code-block:: bash
    	
   	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
        
After installing the package manager you need to activate conda and create a custom environment to run your code on.

.. code-block:: bash
    	
  	source ~/miniconda3/bin/activate
    conda create -n fenics-env -c conda-forge fenics python=3.9
    conda activate fenics-env

**Remember to activate both conda and fenics-env everytime you want to use the code**

`As to why you need conda or a seperate environment at all, read through <https://www.anaconda.com/docs/tools/working-with-conda/environments>`_.

Installing python packages
--------------------------

.. code-block:: bash

    pip install -r requirements.txt

In case the above code returns an error of not being able to find **desist**, then do the following.
The package *desist* isn't a standard built-in python package and therefore needs to be included manually. 
`More information about the *desist* package <https://github.com/UvaCsl/desist>`_.

1. Replace the first line **desist==0.1.0** in the 'requirements.txt' file with **git+https://github.com/UvaCsl/desist.git#egg=desist**
2. Run the above command again. And to check that the installed package and its version.

.. code-block:: bash
    	
  	pip install -r requirements.txt
  	pip show desist

Running the perfusion code
---------------------------


.. code-block:: bash

	cd ~/perfusion_and_tissue_damage/perfusion/
	chmod +x perfusion_runner.sh
	./perfusion_runner.sh
	
The ``perfusion_runner.sh`` needs to be set as an executable file for it to run successfully, and the second line makes sure of that. The ``README.md`` file in the perfusion folder gives a neat description of what the ``perfusion_runner.sh`` does.
