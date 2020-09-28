import numpy as np
import os

n_sampl = 11

os.chdir('../perfusion/')
# os.system('ls')

os.system("mpirun -n 6 python3 permeability_initialiser.py")

for i in range(n_sampl):
    os.system("mpirun -n 6 python3 basic_flow_solver.py --config_file '../sensitivity/config_healthy"+'{:02d}'.format(i)+".yml'")
    os.system("mpirun -n 6 python3 basic_flow_solver.py --config_file '../sensitivity/config_RMCA_occl"+'{:02d}'.format(i)+".yml'")