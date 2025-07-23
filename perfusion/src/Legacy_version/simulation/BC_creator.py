"""
This file reads the brain mesh and creates boundary conditions by distributing
the average volumetric flow rate of blood to the brain on the boundaries based
on their surface area

@author: Tamas Istvan Jozsa
"""
# Python imports
from dolfin import *
import numpy as np
import argparse
import os

# Local imports
from ..io import IO_fcts


def create_BC_creator_parser():
    """
    Create an argument parser for configuring boundary condition creation.

    Returns:
        argparse.ArgumentParser
            Configured argument parser with options for healthy/occluded cases,
            occluded artery IDs, and configuration file path.
    """
    parser = argparse.ArgumentParser(description="Impose boundary conditions without considering large arteries")
    parser.add_argument("--healthy", help="indicate if the case is healthy, used only in healthy cases (optional)",
                        type=bool, default=True)
    parser.add_argument('--occluded', help="must be used in occluded cases", dest='healthy', action='store_false')
    parser.add_argument("--occl_ID", help="a list of integers containing occluded major cerebral artery IDs (e.g. 24 25 26)",
                        nargs='+', type=int, default=[25])
    parser.add_argument("--config_file", help="path to configuration file (string)",
                        type=str, default='./configs/config_basic_flow_solver.yaml')
    return parser


def count_surface_area(mesh, boundary_labels, boundaries, mask):
    """
    Compute the surface area for each labeled boundary region.

    Args:
        mesh : dolfin.Mesh
            The mesh on which boundaries are defined.
        boundary_labels : list or array-like
            List of boundary IDs.
        boundaries : dolfin.MeshFunction
            Mesh function marking boundary subdomains.
        mask : array-like of bool
            Boolean mask indicating which boundary labels to include (e.g., labels > 2).

    Returns:
        surface_area : list of float
            Surface area of each boundary region.
        total_surface_area : float
            Sum of surface areas where mask is True.
    """
    surface_area = []
    dS = ds(subdomain_data=boundaries)
    n_labels = len(boundary_labels)
    for i in range(n_labels):
        ID = int(boundary_labels[i])
        area_value = assemble(Constant(1) * dS(ID, domain=mesh))
        surface_area.append(area_value)
    return surface_area, sum(mask * surface_area)


def create_boundary_values_and_map(mesh_file_split):
    """
    Load and process the boundary mapper CSV to create mapping of boundary labels to indices.

    Args:
        mesh_file_split : list of str
            Result of mesh file path split; mesh_file_split[0] should be the directory path.

    Returns:
        boundary_map : np.ndarray
            Array mapping boundary values to indices (or cluster IDs).
        boundary_values : np.ndarray
            Filtered boundary labels (greater than 2), used for mapping.
    """
    # Load the boundary_mapper.csv
    boundary_mapper = np.loadtxt(mesh_file_split[0] + '/boundary_mapper.csv', skiprows=1, delimiter=',')

    # Load the boundary_mapper.csv (skipping the header)
    boundary_values = np.array(list(set((boundary_mapper[:, 1] > 2) * boundary_mapper[:, 1]))[1::], dtype=int)
    boundary_map = np.zeros(len(boundary_values))

    # Update the map
    counter = 0
    for i in list(boundary_values):
        boundary_map[counter] = int(boundary_mapper[np.argwhere(boundary_mapper[:, 1] == int(i))[0], 0])
        counter = counter + 1

    return boundary_map, boundary_values


def populate_boundary_matrix_test(mask, boundary_labels, boundary_map, boundary_values, Q, p, is_clustered):
    """
    Create the boundary matrix for flow and pressure conditions at surface regions.

    Args:
        mask : array-like of bool
            Indicates which boundary labels are active (> 2).
        boundary_labels : list or np.ndarray
            Labels identifying boundary regions.
        boundary_map : np.ndarray
            Mapping of boundary values to cluster IDs or other identifiers.
        boundary_values : np.ndarray
            Unique boundary labels used for matching.
        Q : np.ndarray
            Flow rate values assigned to each boundary.
        p : np.ndarray
            Pressure values assigned to each boundary.
        is_clustered : bool
            Whether the mesh is clustered (affects whether mapping is applied).

    Returns:
        np.ndarray
            Boundary matrix of shape (N, 5), where each row contains:
            [label, Q, p, mapped_value, flag]
    """
    boundary_matrix = []

    n_mask = len(mask)
    for i in range(n_mask):
        if mask[i] > 0:
            mapped_value = 0
            label = boundary_labels[i]
            flow = Q[i]
            pressure = p[i]

            if is_clustered:
                # Find matching index in boundary_values
                match_indices = np.argwhere(boundary_values == label)
                mapped_value = boundary_map[match_indices[0]] if match_indices.size > 0 else 0

            boundary_matrix.append([label, flow, pressure, mapped_value, 0])

    return np.array(boundary_matrix)


def adjust_boundaries_conditions_with_occlusion(physical_configs, boundary_matrix, occluded_ID):
    """
    Modify the boundary matrix to account for occluded arteries by zeroing flow
    and setting pressure to venous level.

    Args:
        physical_configs : dict
            Dictionary containing physical parameters, must include 'p_venous'.
        boundary_matrix : np.ndarray
            Matrix of boundary conditions.
        occluded_ID : list of int
            List of IDs corresponding to occluded arteries.

    Returns:
        np.ndarray
            Modified boundary matrix with occluded rows updated.
    """
    # Find occluded elements of the matrix
    occluded_ID_set = set(occluded_ID)  # Fasten the process
    mask = np.isin(boundary_matrix[:, 3], list(occluded_ID_set))

    # Update value where occluded
    boundary_matrix[mask, 1] = 0
    boundary_matrix[mask, 2] = physical_configs.get('p_venous', 0)
    boundary_matrix[mask, 4] = 1

    return boundary_matrix


def save_boundary_conditions(result_file, boundary_matrix):
    """
    Save the boundary matrix to a CSV file.

    Args:
        result_file : str
            Path to the output CSV file.
        boundary_matrix : np.ndarray
            Matrix of boundary conditions to be saved. Each row includes:
            [label, Q, p, mapped_value, occlusion_flag]
    """
    fheader = 'cluster ID,Q [ml/s],p [Pa],feeding artery,BC: p->0 or Q->1'
    np.savetxt(result_file, boundary_matrix, "%d,%f,%f,%d,%d", header=fheader)


def main():
    # Extract arguments
    parser = create_BC_creator_parser()
    args = parser.parse_args()

    is_healthy = args.healthy
    occluded_ID = args.occl_ID
    config_file = args.config_file

    # Get basic configurations
    configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
    mesh_file = configs['input']['mesh_file']
    result_file = configs['input']['inlet_boundary_file']
    physical_configs = configs['physical']

    if not os.path.exists(result_file.rsplit('/', 1)[0]):
        os.makedirs(result_file.rsplit('/', 1)[0])

    # Read mesh
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(mesh_file)

    # Volumetric flow rate to the brain [ml / s]
    Q_brain = 10.0

    # Compute surface area for each boundary region
    boundary_labels = list(set(boundaries.array()))
    # 0: interior face, 1: brain stem cut plane,
    # 2: ventricular surface, 2+: brain surface
    # Count superficial regions
    mask = np.array(boundary_labels) > 2
    surface_area, total_surface_area = count_surface_area(mesh, boundary_labels, boundaries, mask)

    # Define volumetric flow rates proportional to superficial surface areas and pressure
    Q = mask * surface_area * Q_brain / total_surface_area
    p = mask * physical_configs.get('p_arterial')

    mesh_file_split = mesh_file.rsplit('/', 1)
    is_clustered = mesh_file_split[-1] == 'clustered.xdmf'

    # Populate the boundary conditions matrix
    if is_clustered:
        boundary_map, boundary_values = create_boundary_values_and_map(mesh_file_split)

    boundary_matrix = populate_boundary_matrix_test(mask, boundary_labels, boundary_map, boundary_values, Q, p, is_clustered)

    # Handle occluded scenario
    if not is_healthy:
        adjust_boundaries_conditions_with_occlusion(physical_configs, boundary_matrix, occluded_ID)

    # Save file
    save_boundary_conditions(result_file, boundary_matrix)


if __name__ == "__main__":
    main()
