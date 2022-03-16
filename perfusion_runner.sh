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
    mpirun -n 6 python3 permeability_initialiser.py --config_file ./config_examples/config_permeability_initialiser4testing.yaml
fi

python3 BC_creator.py
mpirun -n 6 python3 basic_flow_solver.py
python3 convert_res2img.py --config_file ../VP_results/p0000/perfusion_healthy/settings.yaml

mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_LMCAo.yaml
python3 convert_res2img.py --config_file ../VP_results/p0000/perfusion_LMCAo/settings.yaml
python3 lesion_comp_from_img.py --healthy_file ../VP_results/p0000/perfusion_healthy/perfusion.nii.gz --occluded_file ../VP_results/p0000/perfusion_LMCAo/perfusion.nii.gz
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_basic_flow_solver_LMCAo.yaml --baseline ../VP_results/p0000/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0000/perfusion_LMCAo/perfusion.xdmf

mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_basic_flow_solver_RMCAo.yaml
python3 convert_res2img.py --config_file ../VP_results/p0000/perfusion_RMCAo/settings.yaml
python3 lesion_comp_from_img.py --healthy_file ../VP_results/p0000/perfusion_healthy/perfusion.nii.gz --occluded_file ../VP_results/p0000/perfusion_RMCAo/perfusion.nii.gz
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_basic_flow_solver_RMCAo.yaml --baseline ../VP_results/p0000/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0000/perfusion_RMCAo/perfusion.xdmf

# run tissue health model
cd ../tissue_health/
python3 tissue_health_propagation.py --res_yaml ../VP_results/p0000/tissue_damage_RMCAo/infarct.yaml
python3 tissue_health_propagation.py --res_yaml ../VP_results/p0000/tissue_damage_LMCAo/infarct.yaml --config_file ./config_propagation_LMCAo.yaml

