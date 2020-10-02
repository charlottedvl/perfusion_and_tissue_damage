START_TIME=$SECONDS

# generate config files with different input parameters
python3 perfusion_parameter_sampling.py

# generate BC files for healthy and occluded scenarios
python3 ../perfusion/BC_creator.py --res_fldr '../sensitivity/' --config_file '../sensitivity/config_files/config_healthy00.yml'
python3 ../perfusion/BC_creator.py --res_fldr '../sensitivity/' --config_file '../sensitivity/config_files/config_RMCA_occl00.yml' --occluded

# run simulations for each input parameter file
python3 param_mapping_runner.py

# plot infarcted volume as function of input parameters
python3 plot_sensitivity_results.py

# report execution time
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"

# play noise when ready
play -nq -t alsa synth 1 sine 440
