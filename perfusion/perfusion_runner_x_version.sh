if [ -d "./brain_meshes" ]
then
    echo "The archive of brain_meshes has been extracted already"
else
    echo "The archive of brain_meshes will be extracted"
    tar xf ../brain_meshes.tar.xz
fi

if [ -e "./brain_meshes/b0000/permeability/K1_form.h5" ]
then
    echo "The permeability tensor form has been computed already"
else
    echo "The permeability tensor form will be computed"
    mpirun -n 6 python3 -m src.X_version.simulation.permeability_initialiser_x
fi

echo "The basic flow solver is running"
python3 -m src.X_version.simulation.BC_creator_x
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver.yaml
#python3 -m src.X_version.io.convert_res2img --config_file ./results/p0000/perfusion_healthy/settings.yaml

echo "The LMCAo model is running"
python3 -m src.X_version.simulation.BC_creator_x --occl_ID 22 --config_file ./configs/config_basic_flow_solver_LMCAo.yaml
#python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_LMCAo.yaml
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_LMCAo.yaml
#python3 -m src.X_version.io.convert_res2img --config_file ./results/p0000/perfusion_LMCAo/settings.yaml
#python3 -m src.X_version.simulation.lesion_comp_from_img --healthy_file ./results/p0000/perfusion_healthy/perfusion.nii.gz --occluded_file ./results/p0000/perfusion_LMCAo/perfusion.nii.gz
#mpirun -n 6 python3 -m src.X_version.simulation.infarct_calculation_thresholds --config_file ./configs/config_basic_flow_solver_LMCAo.yaml --baseline ./results/p0000/perfusion_healthy/perfusion.xdmf --occluded ./results/p0000/perfusion_LMCAo/perfusion.xdmf

echo "The RMCAo model is running"
python3 -m src.X_version.simulation.BC_creator_x --config_file ./configs/config_basic_flow_solver_RMCAo.yaml
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_RMCAo.yaml
#python3 -m src.X_version.io.convert_res2img --config_file ./results/p0000/perfusion_RMCAo/settings.yaml
#python3 -m src.X_version.simulation.lesion_comp_from_img --healthy_file ./results/p0000/perfusion_healthy/perfusion.nii.gz --occluded_file ./results/p0000/perfusion_RMCAo/perfusion.nii.gz
#mpirun -n 6 python3 -m src.X_version.simulation.infarct_calculation_thresholds --config_file ./configs/config_basic_flow_solver_RMCAo.yaml --baseline ./results/p0000/perfusion_healthy/perfusion.xdmf --occluded ./results/p0000/perfusion_RMCAo/perfusion.xdmf

echo "The LACAo model is running"
python3 -m src.X_version.simulation.BC_creator_x --config_file ./configs/config_basic_flow_solver_LACAo.yaml --occl_ID 21
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_LACAo.yaml

echo "The RACAo model is running"
python3 -m src.X_version.simulation.BC_creator_x --config_file ./configs/config_basic_flow_solver_RACAo.yaml --occl_ID 24
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_RACAo.yaml

echo "The LPCAo model is running"
python3 -m src.X_version.simulation.BC_creator_x --config_file ./configs/config_basic_flow_solver_LPCAo.yaml --occl_ID 23
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_LPCAo.yaml

echo "The RPCAo model is running"
python3 -m src.X_version.simulation.BC_creator_x --config_file ./configs/config_basic_flow_solver_RPCAo.yaml --occl_ID 26
mpirun -n 6 python3 -m src.X_version.simulation.basic_flow_solver_x --config_file ./configs/config_basic_flow_solver_RPCAo.yaml