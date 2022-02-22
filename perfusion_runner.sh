if [ -d "./brain_meshes" ] 
then
    echo "The archive of brain_meshes has be extracted already"
else
    echo "The archive of brain_meshes will be extracted" 
    tar xf brain_meshes.tar.xz
fi

cd perfusion
if [ -e "../brain_meshes/b0000/permeability/K1_form.h5" ] 
then
    echo "The permeability tensor form has been computed already"
else
    echo "The permeability tensor form will be computed" 
    mpirun -n 6 python3 permeability_initialiser.py
fi

python3 BC_creator.py
mpirun -n 6 python3 basic_flow_solver.py
python3 convert_res2img.py --res_fldr ../VP_results/p0000/perfusion_healthy/
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_LMCAo.yaml
python3 convert_res2img.py --res_fldr ../VP_results/p0000/perfusion_LMCAo/
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_RMCAo.yaml
python3 convert_res2img.py --res_fldr ../VP_results/p0000/perfusion_RMCAo/

mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_basic_flow_solver_LMCAo.yaml --baseline ../VP_results/p0000/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0000/perfusion_LMCAo/perfusion.xdmf
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_basic_flow_solver_RMCAo.yaml --baseline ../VP_results/p0000/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0000/perfusion_RMCAo/perfusion.xdmf
