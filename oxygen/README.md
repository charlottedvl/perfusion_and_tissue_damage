# oxygen_main

Runing the 3D oxygen solver:

1; Follow instructions to run the 3D blood flow solver (perfusion_runner.sh) to obtain the inputs for the oxygen model.

2; Compute oxygen concentration distribution with oxygen_main.py
   For parallel execution, use:
   mpirun -n #number_of_processors python3 oxygen_main.py
   Using 6 cores and first order finite elements, the excution is slightly over 2 minutes.
