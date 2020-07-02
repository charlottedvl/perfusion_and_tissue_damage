import os
os.chdir('./perfusion')
os.system('ls -lha')
os.system('pwd')
if not os.path.exists('../brain_meshes/b0000/permeability/K1_form.xdmf'):
    os.system('mpirun --allow-run-as-root -n 6 python3 permeability_initialiser.py')
    
os.system('python3 BC_creator.py')
os.system('mpirun --allow-run-as-root -n 6 python3 basic_flow_solver.py')
