"""
Multi-compartment Darcy flow model with mixed Dirichlet and Neumann
boundary conditions

System of equations (no summation notation)
Div ( Ki Grad(pi) ) - Sum_j=1^3 beta_ij (pi-pj) = sigma_i

Ki - permeability tensor [mm^3 s / g]
pi & pj - volume averaged pressure in the ith & jth comparments [Pa]
beta_ij - coupling coefficient between the ith & jth compartments [Pa / s]
sigma_i - source term in the ith compartment [1 / s]

@author: Tamas Istvan Jozsa
"""

import argparse
import time

import numpy as np
import yaml
from dolfin import *

from ..io import IO_fcts
from ..utils import finite_element_fcts as fe_mod
from ..utils import suppl_fcts, config_utils

# Global settings
np.set_printoptions(linewidth=200)
parameters['ghost_mode'] = 'none'  # ghost mode options: 'none', 'shared_facet', 'shared_vertex'
# solver runs is "silent" mode
set_log_level(50)


def create_infarct_calculation_parser():
    """
    Create a command-line argument parser for infarct volume calculation.

    Returns:
        argparse.ArgumentParser: Configured parser object with all expected arguments:
            - config_file: Path to YAML config file
            - res_fldr: Path to the results folder
            - mesh_file: Path to the brain mesh file
            - inlet_boundary_file: Path to inlet boundary marker file
            - baseline: Path to baseline (healthy) perfusion results
            - occluded: Path to occluded (stroke) perfusion results
            - thresholds: Number of thresholds to use in infarct volume analysis
    """
    parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
    parser.add_argument("--config_file", help="path to configuration file (string ended with /)",
                        type=str, default='./configs/config_coupled_flow_solver.yaml')
    parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                        type=str, default=None)
    parser.add_argument("--mesh_file", help="path to mesh_file",
                        type=str, default=None)
    parser.add_argument("--inlet_boundary_file", help="path to inlet_boundary_file",
                        type=str, default=None)
    parser.add_argument("--baseline", help="path to perfusion output of baseline scenario",
                        type=str, default=None)
    parser.add_argument("--occluded", help="path to perfusion output of stroke scenario",
                        type=str, default=None)
    parser.add_argument("--thresholds", help="number of thresholds to evaluate",
                        type=int, default=21)
    return parser


def compute_infarct_volume_thresholds(thresholds, perfusion_change, K2_space, brain_mesh, subdomains, target=-70):
    """
    Compute infarct volumes across a range of perfusion thresholds.

    Ensures the target threshold (default -70%) is included.
    For each threshold, computes a binary infarct region and evaluates its volume.

    Args:
        thresholds (np.ndarray): Array of perfusion drop thresholds in [%].
        perfusion_change (Function): FEniCS function of relative perfusion change.
        K2_space (FunctionSpace): Function space for projecting binary infarct results.
        brain_mesh (Mesh): The computational mesh.
        subdomains (MeshFunction): Subdomain labels used for volume aggregation.
        target (float, optional): Specific threshold to ensure inclusion (default: -70).

    Returns:
        np.ndarray: Array of shape (N, 4) where each row contains:
                    [threshold, volume_id, total_volume_mm3, infarct_volume_ml].
    """
    # For now a value of `-70%` is assumed as a desired threshold value to determine
    # infarct volume from perfusion data.
    # We ensure that `-70%` is present within the considered threshold values
    if target not in thresholds:
        # [::-1] to reverse sort direction, maintain descending order
        thresholds = np.sort(np.append(thresholds, target))[::-1]

    volume_infarct_values_thresholds = np.empty((0, 4), float)
    # Compute infarct for each threshold
    for threshold in thresholds:
        infarct = project(conditional(gt(perfusion_change, Constant(threshold)), Constant(0.0), Constant(1.0)),
                          K2_space,
                          solver_type='bicgstab', preconditioner_type='petsc_amg')
        infarct_volume = suppl_fcts.infarct_vol(brain_mesh, subdomains, infarct)
        volume_infarct_values = np.concatenate(
            (np.array([threshold, threshold, threshold])[:, np.newaxis], infarct_volume),
            axis=1)
        volume_infarct_values_thresholds = np.append(volume_infarct_values_thresholds, volume_infarct_values, axis=0)
    return volume_infarct_values_thresholds


def save_infarct_calculation_results(result_folder, volume_infarct_values_thresholds):
    """
    Save computed infarct volume data to CSV and YAML summary file.

    Outputs a CSV file with infarct volume per region and threshold.
    Also writes a YAML summary for the infarct volume at -70% perfusion drop
    in the GM+WM region (assumed to be label 23).

    Args:
        result_folder (str): Directory path where output files will be saved.
        volume_infarct_values_thresholds (np.ndarray): Infarct data computed from thresholds,
            with shape (N, 4): [threshold, volume_id, volume_mm3, infarct_volume_ml].

    Side Effects:
        - Writes 'volume_infarct_values_thresholds.csv' to disk.
        - Appends 'perfusion_outcome.yml' with the 30% rCBF core volume.
    """
    file_header = 'threshold [%],volume ID,Volume [mm^3],infarct volume [mL]'
    np.savetxt(result_folder + 'volume_infarct_values_thresholds.csv',
               volume_infarct_values_thresholds, "%e,%d,%e,%e", header=file_header)

    with open(result_folder + "perfusion_outcome.yml", 'a') as outfile:
        # select the row where the perfusion drop is -70%, and the total volume (GM+WM, labelled 23)
        selected_row = np.where((volume_infarct_values_thresholds[:, 0] == -70) &
                                (volume_infarct_values_thresholds[:, 1] == 23))
        volume = volume_infarct_values_thresholds[selected_row, 3]
        yaml.safe_dump(
            {'core-volume_30%_rCBF_mL': float(volume)},
            outfile
        )


def main():
    # Define MPI variables
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print('Step 1: Reading input files, initialising functions and parameters')

    # Get configuration
    parser = create_infarct_calculation_parser()
    args = parser.parse_args()
    config_file = args.config_file
    configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
    result_folder = configs['output']['res_fldr']

    # Define simulation parameters
    simulation_configs = configs.get('simulation', {})
    compartmental_model, velocity_order = config_utils.prepare_simulation_parameters(simulation_configs)

    # Read mesh
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

    # Determine functions spaces
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'], model_type=compartmental_model,
                                vel_order=velocity_order)

    healthy_file, occluded_file = config_utils.extract_files_from_config(args, result_folder)

    if rank == 0:
        print('Step 2: Reading perfusion files')

    # Load previous results
    perfusion = IO_fcts.xdmf_reader(healthy_file, K2_space, 'perfusion')
    perfusion_occluded = IO_fcts.xdmf_reader(occluded_file, K2_space, 'perfusion')

    if rank == 0:
        print('Step 3: Calculating change in perfusion and infarct volume')

    # Calculate change in perfusion and infarct
    perfusion_change = project(((perfusion - perfusion_occluded) / perfusion) * -100, K2_space, solver_type='bicgstab',
                               preconditioner_type='petsc_amg')

    # Compute the infarct volume for different thresholds
    thresholds = np.linspace(0, -100, args.thresholds)
    volume_infarct_values_thresholds = compute_infarct_volume_thresholds(thresholds, perfusion_change, K2_space, mesh,
                                                                         subdomains)

    # Save results
    if rank == 0:
        save_infarct_calculation_results(result_folder, volume_infarct_values_thresholds)
    return


if __name__ == "__main__":
    main()
