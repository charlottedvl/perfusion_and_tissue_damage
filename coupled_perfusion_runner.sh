# this shell script assumes that ../1d-blood-flow exists and the corresponding pythong modules have been insutalled. To this end execute 'sudo python3 setup.py install' in the ../1d-blood-flow folder.

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


cp -r ../../1d-blood-flow/Generated_Patients/patient_0 ./patient_0
cp ../brain_meshes/b0000/clustered* ./patient_0/bf_sim/

mpirun -n 6 python3 coupled_flow_solver.py
mpirun -n 6 python3 coupled_flow_solver.py
