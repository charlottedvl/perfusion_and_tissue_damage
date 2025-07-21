Ares
====

You can access Ares resources with the following command, adjust to your credentials:

.. code-block:: bash

   ssh  your_plg_username@ares.cyfronet.pl

You will be prompted to enter your plgGrid password. Once entered, you'll access the resources.

Initial setup
-------------

Before being able to run the code on Ares, some preliminary steps need to be carried out.
This initial setup is essential to run the code and get inform of the state of each task you submit.

The first step is to upload the code into Ares. To do so, you can choose either of the following solutions: cloning
from the public GitHub repository or copying the folder from your local machine using a command.

To clone the repository, you can use the following command:

.. code-block:: bash

    git clone https://github.com/Gemini-DTH/perfusion_and_tissue_damage.git

If you have already cloned the repository on your local machine, you may want to copy this folder into Crescent2. You can
use the following command to do so:

.. code-block:: bash

    scp -r folder_to_copy/ your_plg_username@ares.cyfronet.pl:~/folder_where_to_copy

Once this is done, you should see the folder in your Ares environment.

The next step is to create or import the container you will be using to run the code. If you have already created it on
your local machine, you can import it using the same command as before:

.. code-block:: bash

    scp -r container.sif your_plg_username@ares.cyfronet.pl:~/folder_where_to_copy

If not, it can be build using a submission file. Ares login nodes don't support apptainer or singularity, but the
compute nodes do. This step can take quite some time, up to 10 to 15 minutes.

The initial setup is now finished. Those steps will not be to be redone.

Running the code
----------------

Everytime you want to run the code, you need to follow those steps. From the root folder, you can submit your job
from the root folder of the project using:

.. code-block:: bash

   	sbatch your_submission_file.sh

To see the state of your job, you can use:

.. code-block:: bash

    squeue

If something seems not right, you can cancel your job by replacing your job ID in:

.. code-block:: bash

    scancel your_job_id
