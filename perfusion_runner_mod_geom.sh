if [ -d "./brain_meshes" ] 
then
    echo "The archive of brain_meshes has be extracted already"
else
    echo "The archive of brain_meshes will be extracted" 
    tar xf brain_meshes.tar.xz
fi

# test virtual patients with transformed brain geometries
python3 VP_mesh_prep.py --age 81.44 --sex 2
python3 VP_mesh_prep.py --age 47.71 --sex 1
cd perfusion/
if [ -e "../brain_meshes/b0000_age81.44_sex2/permeability/K1_form.h5" ] 
then
    echo "The permeability tensor form has been computed already"
else
    echo "The permeability tensor form will be computed" 
    mpirun -n 6 python3 permeability_initialiser.py --config_file ./config_examples/config_permeability_initialiser_mod_geom.yaml
fi
if [ -e "../brain_meshes/b0000_age47.71_sex1/permeability/K1_form.h5" ] 
then
    echo "The permeability tensor form has been computed already"
else
    echo "The permeability tensor form will be computed" 
    mpirun -n 6 python3 permeability_initialiser.py --config_file ./config_examples/config_permeability_initialiser_mod_geom2.yaml
fi

# patient 1 - simulations
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_examples/config_basic_flow_solver_mod_geom.yaml
python3 convert_res2img.py --res_fldr ../VP_results/p0001/perfusion_healthy/
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_examples/config_basic_flow_solver_RMCAo_mod_geom.yaml
python3 convert_res2img.py --res_fldr ../VP_results/p0001/perfusion_RMCAo/
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_examples/config_basic_flow_solver_RMCAo_mod_geom.yaml --baseline ../VP_results/p0001/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0001/perfusion_RMCAo/perfusion.xdmf

# patient 2 - simulations
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_examples/config_basic_flow_solver_mod_geom2.yaml
python3 convert_res2img.py --res_fldr ../VP_results/p0002/perfusion_healthy/
mpirun -n 6 python3 basic_flow_solver.py --config_file ./config_examples/config_basic_flow_solver_RMCAo_mod_geom2.yaml
python3 convert_res2img.py --res_fldr ../VP_results/p0002/perfusion_RMCAo/
mpirun -n 6 python3 infarct_calculation_thresholds.py --config_file ./config_examples/config_basic_flow_solver_RMCAo_mod_geom2.yaml --baseline ../VP_results/p0002/perfusion_healthy/perfusion.xdmf --occluded ../VP_results/p0002/perfusion_RMCAo/perfusion.xdmf
