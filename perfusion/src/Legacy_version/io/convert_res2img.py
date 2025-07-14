"""
This file converts a finite element results (a .h5 or .xdmf) file into a .nii.gz format.

@author: Charlotte Devillé
"""

# Python imports
import dolfin
import sys
import argparse
import numpy
import nibabel as nib
import matplotlib.pyplot as plt
import os

# Local imports
from ..io import IO_fcts
from ..utils import finite_element_fcts as fe_mod


def create_parser():
    """
    Creates and configures an argparse.ArgumentParser for the script.

    This parser defines command-line arguments for converting finite element
    results into NIfTI image files.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert finite elements results (*.h5 and *.xdmf) into an image (*.nii.gz)")
    parser.add_argument("--config_file", help="Path to configuration file",
                        type=str, default='./configs/config_basic_flow_solver.yaml')
    parser.add_argument("--res_fldr", help="Path to results folder (string ended with /)", type=str, default=None)
    parser.add_argument("--variable", help="One of the following: press1, press2, press3, vel1, vel2, vel3, \
                     k1, k2, k3, beta12, beta23, perfusion", type=str, default='perfusion')
    parser.add_argument("--voxel_size", help="Voxel edge size in [mm]", type=int, default=2)
    parser.add_argument("--background_value", help="Value used for background voxels", type=int, default=-1024)
    parser.add_argument('--save_figure', action='store_true', help="Save figure showing image along midline slices", \
                        default=False)
    return parser


def prepare_voxel_size(arguments):
    """
    Prepares the voxel size to be a 3-element NumPy array of floats.

    If the input voxel size from arguments is a single scalar, it is
    replicated across all three dimensions (x, y, z).

    Args:
        arguments (argparse.Namespace): The parsed command-line arguments,
            expected to have a 'voxel_size' attribute.

    Returns:
        numpy.ndarray: A 3-element NumPy array representing the voxel size
                       along x, y, and z axes (e.g., [2.0, 2.0, 2.0]).
    """
    voxel_size = numpy.array(arguments.voxel_size)
    try:
        if len(voxel_size) != 3:
            voxel_size = numpy.array([voxel_size[0], voxel_size[0], voxel_size[0]], dtype=float)
    except:
        voxel_size = numpy.array([voxel_size, voxel_size, voxel_size], dtype=float)
    return voxel_size


def allocate_field(variable_name, mesh, K1_space, K2_space, Vvel, simulation_configs):
    """
    Allocates the appropriate dolfin.Function and determines its type
    based on the variable name prefix.

    Args:
        variable_name (str): The name of the variable (e.g., 'press1', 'vel1', 'perfusion').
        mesh (dolfin.Mesh): The DOLFIN mesh object.
        K1_space (dolfin.FunctionSpace): Function space for K1 (tensor) variables.
        K2_space (dolfin.FunctionSpace): Function space for K2 (scalar, usually related to perfusion/beta) variables.
        Vvel (dolfin.FunctionSpace): Function space for velocity (vector) variables.
        simulation_configs (dict): A dictionary containing simulation configuration settings,
            expected to have 'fe_degr' under 'simulation' key for pressure spaces.

    Returns:
        tuple: A tuple containing:
            - dolfin_function (dolfin.Function): The allocated DOLFIN function object.
            - variable_type (str): The determined type of the variable ('scalar', 'vector', or 'tensor').
            - variable_name (str): The potentially modified variable name (e.g., 'k1' becomes 'K1').
    Raises:
       ValueError: If the variable name prefix is unknown.
    """
    prefix = variable_name[:3]
    dolfin_function, variable_type = None, ""

    if prefix == 'per':
        dolfin_function = dolfin.Function(K2_space)
        variable_type = 'scalar'
    elif prefix == 'pre':
        Vp = dolfin.FunctionSpace(mesh, "Lagrange", simulation_configs.get('fe_degr'))
        dolfin_function = dolfin.Function(Vp)
        variable_type = 'scalar'
    elif prefix == 'vel':
        dolfin_function = dolfin.Function(Vvel)
        variable_type = 'vector'
    elif variable_name[0] == 'k':
        variable_name = variable_name.upper()
        dolfin_function = dolfin.Function(K1_space)
        variable_type = 'tensor'
    elif prefix == 'bet':
        dolfin_function = dolfin.Function(K2_space)
        variable_type = 'scalar'
    else:
        raise ValueError(f"Unknown variable type: {variable_name}")
    return dolfin_function, variable_type, variable_name


def load_dolfin_data(variable, dolfin_variable, results_folder):
    """
    Loads finite element data for a specified variable from an XDMF file.

    This function attempts to read a checkpoint of the DOLFIN function
    from its corresponding XDMF file.

    Args:
        variable (str): The name of the variable as it appears in the XDMF file
                       (e.g., 'perfusion', 'K1').
        dolfin_variable (dolfin.Function): The DOLFIN function object into which
                       the data will be loaded.
        results_folder (str): The path to the directory containing the XDMF files.

    Returns:
        None: The function modifies `dolfin_variable` in-place.

    Prints:
        A message if the specified XDMF file is not available (due to ValueError).
    """
    try:
        f_in = dolfin.XDMFFile(results_folder + variable + '.xdmf')
        f_in.read_checkpoint(dolfin_variable, variable, 0)
        f_in.close()
    except ValueError:
        print(variable + '.xdmf file not available!')
    return


def create_image_grid(mesh, voxel_size):
    """
    Creates the coordinate arrays for the image grid based on the mesh
    and desired voxel size.

    The image grid extends slightly beyond the mesh's bounding box
    to ensure full coverage.

    Args:
        mesh (dolfin.Mesh): The DOLFIN mesh object from which to determine
                            the bounding box for the image grid.
        voxel_size (numpy.ndarray): A 3-element array specifying the size of
                            each voxel in x, y, and z directions.

    Returns:
        tuple: A tuple containing:
            - image_coord_min (numpy.ndarray): Minimum coordinates of the image grid origin.
            - x (numpy.ndarray): Array of x-coordinates for the grid.
            - y (numpy.ndarray): Array of y-coordinates for the grid.
            - z (numpy.ndarray): Array of z-coordinates for the grid.
    """
    image_coord_min = numpy.int32(numpy.floor(numpy.min(mesh.coordinates(), axis=0) - voxel_size))
    image_coord_max = numpy.int32(numpy.ceil(numpy.max(mesh.coordinates(), axis=0) + voxel_size))

    x = numpy.arange(image_coord_min[0], image_coord_max[0], voxel_size[0])
    y = numpy.arange(image_coord_min[1], image_coord_max[1], voxel_size[1])
    z = numpy.arange(image_coord_min[2], image_coord_max[2], voxel_size[2])
    return image_coord_min, x, y, z


# TODO: speed up image recovery
def finite_element_to_image_data(var, variable_type, x, y, z, length_x, length_y, length_z, bg_value):
    """
    Converts finite element function data into a NumPy image array.

    The function iterates through a 3D grid, evaluates the DOLFIN function
    at each grid point, and stores the result in a NumPy array. Points where
    the evaluation fails (e.g., outside the mesh) are filled with a background value.

    Args:
        var (dolfin.Function): The DOLFIN function object containing the finite element solution.
        variable_type (str): The type of the variable ('scalar', 'vector', or 'tensor').
        x (numpy.ndarray): Array of x-coordinates for the image grid.
        y (numpy.ndarray): Array of y-coordinates for the image grid.
        z (numpy.ndarray): Array of z-coordinates for the image grid.
        length_x (int): Number of voxels in the x-dimension.
        length_y (int): Number of voxels in the y-dimension.
        length_z (int): Number of voxels in the z-dimension.
        bg_value (float or int): The value to use for background voxels or where function evaluation fails.

    Returns:
        numpy.ndarray: The populated image data array with dimensions (length_x,
                       length_y, length_z) for scalar, (length_x, length_y,
                       length_z, 3) for vector, or (length_x, length_y,
                       length_z, 9) for tensor.

    Raises:
        ValueError: If an unknown variable type is provided.
    """
    if variable_type == 'scalar':
        image_data = numpy.ones((length_x, length_y, length_z)) * bg_value
    elif variable_type == 'vector':
        image_data = numpy.ones((length_x, length_y, length_z, 3)) * bg_value
    elif variable_type == 'tensor':
        image_data = numpy.ones((length_x, length_y, length_z, 9)) * bg_value
    else:
        raise ValueError(f"Unknown var_type: {variable_type}")

    for i in range(length_x):
        for j in range(length_y):
            for k in range(length_z):
                point = (x[i], y[j], z[k])
                try:
                    value = var(point)
                    if variable_type == 'scalar':
                        image_data[i, j, k] = value
                    else:
                        image_data[i, j, k, :] = value
                except Exception:
                    pass  # image_date have been instantiated to the default value so no need to update again
    return image_data


def save_nifti(voxel_size, image_coord_min, image_data, results_folder, variable):
    """
    Saves the image data to a NIfTI (.nii.gz) file.

    This function constructs the affine matrix necessary for NIfTI to correctly
    position the image in 3D space, then saves the image.

    Args:
        voxel_size (numpy.ndarray): A 3-element array specifying the size of each voxel in x, y, and z directions.
        image_coord_min (numpy.ndarray): Minimum coordinates of the image grid origin.
        image_data (numpy.ndarray): The 3D or 4D NumPy array containing the image data.
        results_folder (str): The path to the directory where the NIfTI file will be saved.
        variable (str): The name of the variable, used for the output file name.

    Returns:
        None
    """
    affine_matrix = numpy.eye(4)
    for i in range(3): affine_matrix[i, i] = voxel_size[i]
    affine_matrix[:3, -1] = image_coord_min + 1
    img = nib.Nifti1Image(image_data, affine_matrix)
    nib.save(img, results_folder + variable + '.nii.gz')
    return


def save_image(image_data, results_directory, args_variable, length_x, length_y, length_z):
    """
    Generates and saves 2D slices of the image data along midline planes.

    For vector (4D) data, it calculates the L2 norm to convert it to a scalar field
    before slicing. This function only saves images for 3D (scalar) data.

    Args:
        image_data (numpy.ndarray): The 3D or 4D NumPy array containing the image data.
        results_directory (str): The path to the directory where the PNG figure will be saved.
        args_variable (str): The original variable name from arguments, used for naming the output figure.
        length_x (int): Number of voxels in the x-dimension.
        length_y (int): Number of voxels in the y-dimension.
        length_z (int): Number of voxels in the z-dimension.

    Returns:
        None

    Prints:
        A message if saving figures for tensor spaces is not available.
    """
    dimensions = len(list(image_data.shape))
    if dimensions == 4:
        image_data = numpy.linalg.norm(image_data, axis=3)
    if dimensions != 3:
        print('Saving figure is not available for tensor spaces!')
        return
    slices = [image_data[int(length_x / 2), :, :],
              image_data[:, int(length_y / 2), :],
              image_data[:, :, int(length_z / 2)]]

    fsx = 17
    fsy = 8
    fig1 = plt.figure(num=1, figsize=(fsx / 2.54, fsy / 2.54))
    gs1 = plt.GridSpec(1, 2)
    gs1.update(left=0.05, right=0.99, bottom=0.01, top=0.99, wspace=0.2)

    for index in [1, 2]:
        ax = plt.subplot(gs1[0, index - 1])
        ax.imshow(numpy.flip(numpy.rot90(slices[index]), axis=1), cmap='gist_gray', vmin=0, vmax=image_data.max())
    fig1.savefig(results_directory + args_variable.strip().lower() + '.png', transparent=True, dpi=450)


def main():
    # DOLFIN settings
    dolfin.parameters['ghost_mode'] = 'none'  # ghost mode options: 'none', 'shared_facet', 'shared_vertex'
    # solver runs is "silent" mode
    dolfin.set_log_level(50)

    # Read input
    parser = create_parser()
    args = parser.parse_args()

    # Read config file
    config_file = args.config_file
    if not os.path.isfile(config_file):
        config_file = args.res_fldr + 'settings.yaml'

    voxel_size = prepare_voxel_size(args)

    configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
    results_folder = configs['output']['res_fldr']

    # Simulation parameters
    simulation = configs.get('simulation', {})
    compartmental_model = simulation.get('model_type', 'acv').lower().strip()
    velocity_order = simulation.get('vel_order', simulation.get('fe_degr', 2) - 1)

    # Read mesh
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

    # Determine functions space
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, simulation.get('fe_degr'), model_type=compartmental_model,
                                vel_order=velocity_order)

    # Check if variable is a valid variable
    variable = args.variable.strip().lower()
    valid_variables = ['press1', 'press2', 'press3', 'vel1', 'vel2', 'vel3',
                       'k1', 'k2', 'k3', 'beta12', 'beta23', 'perfusion']

    if variable not in valid_variables:
        sys.exit("Variable specified by '--variable' is not available")

    # Allocate field according to the variable studied
    dolfin_variable, var_type, variable = allocate_field(variable, mesh, K1_space, K2_space, Vvel, simulation)

    load_dolfin_data(variable, dolfin_variable, results_folder)

    # Create the coordinates for the image
    img_coord_min, x, y, z = create_image_grid(mesh, voxel_size)
    nx, ny, nz = len(x), len(y), len(z)

    # Convert finite element data to image
    img_data = finite_element_to_image_data(dolfin_variable, var_type, x, y, z, nx, ny, nz, args.background_value)

    # Save image to nifti format
    save_nifti(voxel_size, img_coord_min, img_data, results_folder, variable)

    # Save image slices if possible
    if args.save_figure:
        save_image(img_data, results_folder, args.variable, nx, ny, nz)


if __name__ == "__main__":
    main()
