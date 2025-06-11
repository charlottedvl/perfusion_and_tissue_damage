import numpy as np
import yaml
import os
import argparse

import analyt_fcts


#%% load settings

parser = argparse.ArgumentParser(description="coupled 1D perfusion and network simulator based on analytical solutions")
parser.add_argument("--config_file", help="path to configuration file",
                    type=str, default='./config_coupled_analyt.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default=None)
config_file = parser.parse_args().config_file

with open(config_file, "r") as myconfigfile:
        configs = yaml.load(myconfigfile, yaml.SafeLoader)


#%% specify 1D network and continuum problems
D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw = analyt_fcts.set_up_network(configs)
beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con = analyt_fcts.set_up_continuum(configs)


#%% construct and solve linear equation system, compute results
A = np.eye(Nn+2*Nc)
b = np.zeros([Nn+2*Nc])
analyt_fcts.define_network_eq(configs, A, b, D, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw, beta, x, Nc, subdom_id, D_ave, G)
analyt_fcts.define_continuum_eq(configs, A, b, beta, Nn, Nc, BC_type_con, BC_val_con, G, l_subdom, x)
xvec = np.linalg.solve(A,b)

# broadcast network and continuum solutions
P, Q, p, vel = analyt_fcts.comp_res(configs,beta,subdom_id,xvec,x,G,Nn,Nc,)

# save results
con_data = np.zeros([len(x),3])
con_data[:,0] = x
con_data[:,1] = p
con_data[:,2] = vel

if parser.parse_args().res_fldr != None:
    configs['res_path'] = parser.parse_args().res_fldr
if not os.path.exists(configs['res_path']):
    os.makedirs(configs['res_path'])

np.savetxt(configs['res_path']+'con_data.csv', con_data, delimiter=',')
np.savetxt(configs['res_path']+'P_ntw.csv', P, delimiter=',')
np.savetxt(configs['res_path']+'Q_ntw.csv', Q, delimiter=',')