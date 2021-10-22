python3 gen_verif_files.py
cd ../
python3 basic_flow_solver.py --config_file ./verification/config_basic_flow_solver_verification.yaml
cd ./verification
python3 analyt_coupled_models.py --config_file config_decoupled_analyt.yaml
python3 postproc_analyt.py --config_file config_decoupled_analyt.yaml
python3 comp_analyt_vs_numeric.py
