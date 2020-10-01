# generate config files with different input parameters
python3 perfusion_parameter_sampling.py

# generate BC files for healthy and occluded scenarios
python3 ../perfusion/BC_creator.py --res_fldr '../sensitivity/' --config_file '../sensitivity/config_files/config_healthy00.yml'
python3 ../perfusion/BC_creator.py --res_fldr '../sensitivity/' --config_file '../sensitivity/config_files/config_RMCA_occl00.yml' --occluded

# run simulations for each input parameter file
python3 param_mapping_runner.py
