import numpy as np
import yaml

n_sampl = 11
K1gm_ref_sampl = np.logspace(-4,-2,n_sampl)

config = dict(
    input = dict(
        mesh_file = '../brain_meshes/b0000/clustered.xdmf',
        read_inlet_boundary = True,
        inlet_boundary_file = '../sensitivity/healthy_BCs.csv',
        inlet_BC_type = 'DBC',
        permeability_folder = '../brain_meshes/b0000/permeability/'
    ),
    physical = dict(
        p_arterial = 10000.0,
        p_venous = 0.0,
        K1gm_ref = 0.001234,
        K2gm_ref = 4.28e-7,
        K3gm_ref = 2.468e-3,
        gmowm_perm_rat = 1.0,
        beta12gm = 1.326e-6,
        beta23gm = 4.641e-06,
        gmowm_beta_rat = 2.538,
    ),
    simulation = dict(
        fe_degr = 1
    ),
    output = dict(
        res_fldr = '../VP_results/p0000/perfusion_healthy/',
        save_pvd = False,
        comp_ave = True,
    )
)

for i in range(n_sampl):

    config['physical']['K1gm_ref'] = float( K1gm_ref_sampl[i] )
    
    config['input']['inlet_boundary_file'] = '../sensitivity/healthy_BCs.csv'
    config['input']['inlet_BC_type'] = 'DBC'
    config['output']['res_fldr'] = '../sensitivity/healthy'+'{:02d}'.format(i)+'/'
    
    with open('config_healthy'+'{:02d}'.format(i)+'.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    config['input']['inlet_boundary_file'] = '../sensitivity/RMCA_occl_BCs.csv'
    config['input']['inlet_BC_type'] = 'mixed'
    config['output']['res_fldr'] = '../sensitivity/RMCA_occl'+'{:02d}'.format(i)+'/'
        
    with open('config_RMCA_occl'+'{:02d}'.format(i)+'.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)