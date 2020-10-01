import numpy as np
import yaml
import os

n_sampl = 3

os.chdir('../perfusion/')

if not os.path.exists('../brain_meshes/b0000/permeability/'):
    os.system("mpirun -n 6 python3 permeability_initialiser.py")

os.chdir('../sensitivity/')

for i in range(n_sampl):
    os.system("mpirun -n 6 python3 ../perfusion/basic_flow_solver.py --config_file './config_files/config_healthy"   +'{:02d}'.format(i)+".yml'")
    os.system("mpirun -n 6 python3 ../perfusion/basic_flow_solver.py --config_file './config_files/config_RMCA_occl" +'{:02d}'.format(i)+".yml'")
    
    os.system("mpirun -n 6 python3 infarct_calculation.py    --healthy_config_file './config_files/config_healthy"   +'{:02d}'.format(i)+".yml' --occluded_config_file './config_files/config_RMCA_occl" +'{:02d}'.format(i)+".yml'")
    healthy_config_file  = './config_files/config_healthy'   +'{:02d}'.format(i)+'.yml'
    occluded_config_file = './config_files/config_RMCA_occl' +'{:02d}'.format(i)+'.yml'
    
    with open(healthy_config_file, "r") as configfile:
            healthy_configs = yaml.load(configfile, yaml.SafeLoader)
    with open(occluded_config_file, "r") as configfile:
            occluded_configs = yaml.load(configfile, yaml.SafeLoader)
    
    os.chdir(healthy_configs['output']['res_fldr'])
    os.system('rm *.h5')
    os.system('rm *.xdmf')
    os.chdir('../')
    os.chdir(occluded_configs['output']['res_fldr'])
    os.system('rm *.h5')
    os.system('rm *.xdmf')
    os.chdir('../')