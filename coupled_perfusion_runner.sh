# this shell script assumes that ../bloodflow exists and the corresponding python modules have been installed. To this end execute 'sudo python3 setup.py install' in the ../bloodflow folder.
# Python environment required with all modules for the bloodflow and perfusion models.

if [ -d "./brain_meshes" ] 
then
    echo "The archive of brain_meshes has been extracted already"
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

cd ..
cp -TR ../../bloodflow/DataFiles/DefaultPatient "./patient_0/"
python3 ../../bloodflow/Blood_Flow_1D/GenerateBloodflowFiles.py "./patient_0/"
python3 convert_msh2hdf5.py "./patient_0/bf_sim/clustered_mesh.msh"  "./patient_0/bf_sim/clustered_mesh"

rm -f "./patient_0/bf_sim/Coupled_resistance.csv"
mpirun -n 6 python3 coupled_flow_solver.py
cp -TR  ./patient_0/perfusion "./patient_0/perfusion_healthy"
mpirun -n 6 python3 coupled_flow_solver.py
cp -TR ./patient_0/perfusion "./patient_0/perfusion_stroke"
python3 infarct_calculation_thresholds.py --baseline "./patient_0/perfusion_healthy/perfusion.xdmf" --occluded "./patient_0/perfusion_stroke/perfusion.xdmf"
