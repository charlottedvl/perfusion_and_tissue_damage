python3 gen_verif_files.py

cd verification_coupled
python3 BFOnly.py ./

cd ../../
rm -f ./verification/verification_coupled/bf_sim/Coupled_resistance.csv
mpirun -n 6 python3 coupled_flow_solver.py --config_file ./verification/config_coupled_solver.yaml
cp -TR  ./verification//verification_mesh/results/ ./verification/verification_mesh/results_healthy/
mpirun -n 6 python3 coupled_flow_solver.py --config_file ./verification/config_coupled_solver.yaml

cd ./verification
python3 analyt_coupled_models.py --config_file config_decoupled_analyt.yaml
python3 postproc_analyt.py --config_file config_decoupled_analyt.yaml
python3 comp_analyt_vs_numeric.py
