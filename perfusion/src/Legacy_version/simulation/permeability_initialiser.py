"""
This script generates an anisotropic permeability field based on a predefined
form of the permeability tensor given in a reference coordinate system.

K1_form is the forms of the permeability tensor defined in a rerence
coordinate system, in which e_ref [0,0,1] unit vector is the coordinate
direction perpendicular to the cortical surface.

Then K1_loc is computed as
K1_loc = T*K1_form*T'

T is the transformation matrix handling rotation based on e_ref and e_loc,
where e_loc is the unit vector showing the direction normal to the cortical
surface locally.

e_loc = - grad(pe)/|grad(pe)| where Laplacian(pe) = 0 with the following BCs:
pe = 1 @ cortical surface
pe = 0 @ ventricular surface
d pe / d n = 0 @ brain stem cut plane

@author: Tamas Istvan Jozsa
"""

# Python imports
from dolfin import *
import time
import argparse

# Local imports
from ..io import IO_fcts, permeability_initialiser_IO
from ..utils import suppl_fcts

# Global settings
set_log_level(50)


def create_perm_initialiser_parser():
    """
    Create an argument parser for permeability tensor initialization.

    Returns:
        argparse.ArgumentParser: Parser with an option to set a config file path.
    """
    parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
    parser.add_argument("--config_file", help="path to configuration file",
                        type=str, default='./configs/config_permeability_initialiser.yaml')
    return parser


def save_outputs_perm_initialiser(results_folder, K1, e_loc, main_direction):
    """
    Save key output fields (K1_form, e_loc, main_direction) to XDMF files.

    Args:
        results_folder (str): Path to the output directory.
        K1 (dolfin.Function): Computed permeability tensor field.
        e_loc (dolfin.Function): Local vessel direction vector field.
        main_direction (dolfin.Function): Primary direction vector (for visualization).
    """
    # Save outputs
    # TODO: Compress output and add postprocessing options
    with XDMFFile(results_folder + 'K1_form.xdmf') as file:
        file.write_checkpoint(K1,"K1_form", 0, XDMFFile.Encoding.HDF5, False)

    with XDMFFile(results_folder + 'e_loc.xdmf') as file:
        file.write_checkpoint(e_loc,"e_loc", 0, XDMFFile.Encoding.HDF5, False)

    # Main_direction is non-essential output
    with XDMFFile(results_folder + 'main_direction.xdmf') as file:
        file.write(main_direction)


def save_variables(rank, configs, results_folder, K1, e_loc, main_direction):
    """
    Save output variables as specified in the config file.

    Args:
        rank (int): MPI rank of the process (only rank 0 prints warnings).
        configs (dict): Configuration dictionary loaded from YAML.
        results_folder (str): Directory where results will be saved.
        K1 (dolfin.Function): Permeability tensor field.
        e_loc (dolfin.Function): Local vessel direction vector field.
        main_direction (dolfin.Function): Primary direction vector field.
    """
    myResults = {}
    output_variables = configs['output']['res_vars']
    if len(output_variables) > 0:
        myResults['K1_form'] = K1
        myResults['e_loc'] = e_loc
        myResults['main_direction'] = main_direction
    else:
        if rank == 0: print('No variables have been defined for saving!')

    # Save variables
    res_keys = set(myResults.keys())
    for variable in output_variables:
        if variable in res_keys:
            with XDMFFile(results_folder + variable + '.xdmf') as file:
                file.write_checkpoint(myResults[variable], variable, 0, XDMFFile.Encoding.HDF5, False)
        else:
            if rank==0: print('warning: '+ variable +' variable cannot be saved - variable undefined!')


def saving_results(rank, configs, results_folder, K1, e_loc, main_direction):
    """
    Save computed permeability-related fields using two helper functions.

    Args:
        rank (int): MPI rank of the process.
        configs (dict): Configuration dictionary.
        results_folder (str): Output directory.
        K1 (dolfin.Function): Permeability tensor field.
        e_loc (dolfin.Function): Local vessel orientation vector.
        main_direction (dolfin.Function): Main direction vector.
    """
    save_outputs_perm_initialiser(results_folder, K1, e_loc, main_direction)
    save_variables(rank, configs, results_folder, K1, e_loc, main_direction)


def main():
    # Define MPI variables
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: print('Step 1: Reading input files')

    # Get arguments
    parser = create_perm_initialiser_parser()
    config_file = parser.parse_args().config_file
    configs = permeability_initialiser_IO.permeability_initialiser_config_reader_yml(config_file)
    physical_configs = configs['physical']
    results_folder = configs['output']['res_fldr']

    # Read mesh
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

    # Compute permeability tensor
    if rank == 0: print('Step 2: Computing permeability tensor')

    K_space = TensorFunctionSpace(mesh, "DG", 0)
    e_loc, main_direction = suppl_fcts.comp_vessel_orientation(subdomains, boundaries, mesh, results_folder,
                                                               configs['output']['save_subres'])

    start1 = time.time()
    # Compute permeability tensor
    K1 = suppl_fcts.perm_tens_comp(K_space, subdomains, mesh, physical_configs.get('e_ref'), e_loc,
                                   physical_configs.get('K1_form'))
    end1 = time.time()
    if rank == 0: print("Permeability tensor computation on processor 0 took ", '{:.2f}'.format(end1 - start1),
                        '[s]\n')

    if rank == 0: print('Step 3: Saving output files')
    saving_results(rank, configs, results_folder, K1, e_loc, main_direction)


if __name__ == "__main__":
    main()
