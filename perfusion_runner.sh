if [ -d "./brain_meshes" ] 
then
    echo "The archive of brain_meshes has be extracted already"
else
    echo "The archive of brain_meshes will be extracted" 
    tar xf brain_meshes.tar.xz
fi

cd perfusion
mpirun -n 6 python3 permeability_initialiser.py
mpirun -n 6 python3 basic_flow_solver.py
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_RMCAo.yaml
