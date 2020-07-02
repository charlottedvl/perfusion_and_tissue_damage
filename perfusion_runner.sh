cd perfusion
mpirun -n 6 python3 permeability_initialiser.py
python3 BC_creator.py
mpirun -n 6 python3 basic_flow_solver.py
