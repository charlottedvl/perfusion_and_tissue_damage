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
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_LMCAo.yaml
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_RMCAo.yaml
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_basic_flow_solver_LMCAo.yaml --baseline ../VP_results/p0000/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0000/perfusion_LMCAo/perfusion.xdmf
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_basic_flow_solver_RMCAo.yaml --baseline ../VP_results/p0000/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0000/perfusion_RMCAo/perfusion.xdmf
