# automating the parameter optimiser
import shutil
import yaml
import os
import pandas as pd

# Script to compute all parameters of the perfusion model for different surface pressures and target perfusion levels.
pressures = (10000, 9500, 9000, 8500, 8000, 7500, 7000, 6500, 6000)
mpi = True  # set to False to disable mpi

for pressure in pressures:
    # Assuming total CBF of 600mL/min
    print("Patient")
    V_gm = 894  # mL
    V_wm = 496  # mL
    # P_a = 8000  # Pa
    P_a = pressure  # Pa
    B_ratio = 3.5
    F_ratio = 2.7

    print("Take Q_in as input and calculate parameters")
    Q_in = 600  # mL/min
    F_wm = 100*Q_in / (F_ratio * V_gm + V_wm)
    F_gm = F_ratio * F_wm
    F_B = (F_wm*V_wm+F_gm*V_gm)/(V_wm+V_gm)
    B_ac = (F_gm/60)*0.01 / (P_a - P_a / B_ratio)
    B_cv = B_ratio * B_ac
    print(f"f_gm:{F_gm} ml/min/100mL")
    print(f"f_wm:{F_wm} ml/min/100mL")
    print(f"B_ac:{B_ac} 1/pa/s")
    print(f"B_cv:{B_cv} 1/pa/s")
    print(f"f_B:{F_B} ml/min/100mL")

    ymlfile = "config_coupled_flow_solver.yaml"
    with open(ymlfile, "r") as configfile:
        configs = yaml.load(configfile, yaml.SafeLoader)
    configs['physical']['beta12gm'] = B_ac
    configs['physical']['beta23gm'] = B_cv
    configs['physical']['p_arterial'] = P_a
    configs['optimisation']['FGtarget'] = F_gm
    configs['optimisation']['FWtarget'] = F_wm
    configs['simulation']['fe_degr'] = 1
    configs['input']['read_inlet_boundary'] = False
    with open(ymlfile, 'w') as file:
        yaml.dump(configs, file)

    # run optimiser
    if mpi:
        os.system('mpirun -n 8 python3 parameter_optimiser.py')
    else:
        os.system('python3 parameter_optimiser.py')

    # load optim results
    optim_results = configs['output']['res_fldr'] + "opt_res_Nelder-Mead.csv"

    data = pd.read_csv(optim_results)
    configs['physical']['gmowm_beta_rat'] = float(data["# gmowm_beta_rat"].iloc[-1])
    configs['physical']['K1gm_ref'] = float(data[" K1gm_ref"].iloc[-1])
    configs['physical']['K3gm_ref'] = 2 * float(configs['physical']['K1gm_ref'])
    configs['simulation']['fe_degr'] = 2
    configs['input']['read_inlet_boundary'] = True

    ymlfile = "config_"+str(P_a)+"_Q.yaml"
    with open(ymlfile, 'w') as file:
        yaml.dump(configs, file)

    src_dir = optim_results
    dst_dir = configs['output']['res_fldr']+"opt_res"+str(P_a)+"_Nelder-Mead_Q.csv"
    shutil.copy(src_dir, dst_dir)

##################

for pressure in pressures:
    # assuming total CBF of 50 ml/min/100mL
    print("Patient")
    V_gm = 894  # mL
    V_wm = 496  # mL
    # P_a = 8000  # Pa
    P_a = pressure  # Pa
    B_ratio = 3.5
    F_ratio = 2.7

    print("Take F_B as input and calculate parameters")
    F_B = 50
    F_wm = F_B * (V_wm + V_gm)/(V_wm+F_ratio * V_gm)
    F_gm = F_wm * F_ratio
    B_ac = (F_gm/60)*0.01 / (P_a - P_a / B_ratio)
    B_cv = B_ratio * B_ac

    print(f"f_gm:{F_gm} ml/min/100mL")
    print(f"f_wm:{F_wm} ml/min/100mL")
    print(f"B_ac:{B_ac} 1/pa/s")
    print(f"B_cv:{B_cv} 1/pa/s")
    print(f"f_B:{F_B} ml/min/100mL")

    ymlfile = "config_coupled_flow_solver.yaml"
    with open(ymlfile, "r") as configfile:
        configs = yaml.load(configfile, yaml.SafeLoader)
    configs['physical']['beta12gm'] = B_ac
    configs['physical']['beta23gm'] = B_cv
    configs['physical']['p_arterial'] = P_a
    configs['optimisation']['FGtarget'] = F_gm
    configs['optimisation']['FWtarget'] = F_wm
    configs['simulation']['fe_degr'] = 1
    configs['input']['read_inlet_boundary'] = False

    with open(ymlfile, 'w') as file:
        yaml.dump(configs, file)

    # run optimiser
    if mpi:
        os.system('mpirun -n 8 python3 parameter_optimiser.py')
    else:
        os.system('python3 parameter_optimiser.py')

    # load optim results
    optim_results = configs['output']['res_fldr'] + "opt_res_Nelder-Mead.csv"

    data = pd.read_csv(optim_results)
    configs['physical']['gmowm_beta_rat'] = float(data["# gmowm_beta_rat"].iloc[-1])
    configs['physical']['K1gm_ref'] = float(data[" K1gm_ref"].iloc[-1])
    configs['physical']['K3gm_ref'] = 2 * float(configs['physical']['K1gm_ref'])
    configs['simulation']['fe_degr'] = 2
    configs['input']['read_inlet_boundary'] = True

    ymlfile = "config_"+str(P_a)+"_CBF.yaml"
    with open(ymlfile, 'w') as file:
        yaml.dump(configs, file)

    src_dir = optim_results
    dst_dir = configs['output']['res_fldr']+"opt_res"+str(P_a)+"_Nelder-Mead_CBF.csv"
    shutil.copy(src_dir, dst_dir)
