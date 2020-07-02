# basic_flow_solver

Running the 3D blood flow solver:

1; extract brain_meshes.tar.xz placed in the main repository

2; compute the permeability tensor with permeability_initialiser.py.
For parallel execution, use
mpirun -n #number_of_processors python3 permeability_initialiser.py
Using 4 cores execution takes typically less than 2 minutes.
Parameters are obtained from the config_permeability_initialiser.xml file.
This script has to be executed only once.

3; run the BC_creator.py file (in serial) to generate simple boundary conditions:
python3 BC_creator.py

4; compute the pressure and the velocity field using the basic_flow_solver.py.
The solver can be executed in parallel with
mpirun -n #number_of_processors python3 complex_geom_solver.py
Using 4 cores and first order finite elements, the execution is typically less than 2 minute.
Parameters are obtained from the config_basic_flow_solver.xml file.

