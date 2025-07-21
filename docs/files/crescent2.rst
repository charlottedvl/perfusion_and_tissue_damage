Crescent2
=========

You can access Crescent2 resources with the following command, adjust to your credentials:

.. code-block:: bash

   ssh -X your_crescent2_username@crescent2.central.cranfield.ac.uk

You will be prompted to enter your Crescent2 password. Once entered, you'll access the resources.

Initial setup
-------------

Before being able to run the code on Crescent2, some preliminary steps need to be carried out.
This initial setup is essential to run the code and get inform of the state of each task you submit.

The first step is to upload the code into Crescent2. To do so, you can choose either of thefollowing solutions: cloning
from the public GitHub repository or copying the folder from your local machine using a command.

To clone the repository, you can use the following command:

.. code-block:: bash

    git clone https://github.com/Gemini-DTH/perfusion_and_tissue_damage.git

If you have already cloned the repository on your local machine, you may want to copy this folder into Crescent2. You can
use the following command to do so:

.. code-block:: bash

    scp -r folder_to_copy/ your_crescent2_username@crescent2.central.cranfield.ac.uk:/mnt/beegfs/home/your_crescent2_username/the_folder_you_want_to_copy_in/

Once this is done, you should see the folder in your Crescent2 environment.

The next step is to create or import the container you will be using to run the code. If you have already created it on
your local machine, you can import it using the same command as before:

.. code-block:: bash

    scp -r container.sif your_crescent2_username@crescent2.central.cranfield.ac.uk:/mnt/beegfs/home/your_crescent2_username/you_project_folder/containers/

If not, it can be easily build from the root folder of your project using the following commang:

.. code-block:: bash

    apptainer build containers/container.sif containers/container.def

This step can take quite some time, up to 10 to 15 minutes.

Once the container is built, you will now need to update some scripts. All the submission files need to be updated with
your Cranfield address mail. Go into the folder hpc_submission/Crescent2 and update all the files with the extension
.sub. Please update the following part of the code:

.. code-block:: bash

    ## STEP 4:
    ##
    ## Replace the hpc@cranfield.ac.uk email address
    ## with your Cranfield email address on the #PBS -M line below:
    ## Your email address is NOT your username
    ##
    #PBS -m abe
    #PBS -M your_email@cranfield.ac.uk

This modification ensures that you will be correctly notified for any updates on the submission of your job.

The initial setup is now finished. Those steps will not be to be redone.

Running the code
----------------

Everytime you want to run the code, you need to follow those steps. From the root folder, you can submit your job
from the root folder of the project using:

.. code-block:: bash

   	qsub hpc_submission/Crescent2/your_submission_file.sub

To see the state of your job, you can use:

.. code-block:: bash

    qstat
    # OR
    qstat -a # this will give more information about the running jobs

If something seems not right, you can cancel your job by replacing your job ID in:

.. code-block:: bash

    qdel your_job_id
