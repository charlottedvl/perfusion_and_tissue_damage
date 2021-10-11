cd perfusion

python3 gen_verif_files.py
python3 basic_flow_solver.py --config_file ./config_examples/config_basic_flow_solver_verification.yaml
