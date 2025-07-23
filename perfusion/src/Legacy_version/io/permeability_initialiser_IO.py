import yaml
from dolfin import *
import numpy as np
import untangle


def permeability_initialiser_config_reader(input_file_path):
    """
    Read and parse permeability initialisation configuration from an XML file.

    This function reads an XML-based configuration file using `untangle`, extracts
    relevant values for the permeability tensor initialization process, including:
    mesh path, reference vector, tensor components, result output folder, and
    whether to save subresults.

    Args:
        input_file_path (str): Path to the XML configuration file.

    Returns:
        mesh_file (str): Path to the input mesh file.
        e_ref (np.ndarray): Reference unit vector as a NumPy array of shape (3,).
        K1_form (np.ndarray): Permeability tensor form as a NumPy 3x3 array.
        result_folder (str): Path to the folder for saving results.
        save_subres (bool): Flag indicating whether to save intermediate results.
    """
    configs = untangle.parse(input_file_path).permeability_initialiser

    mesh_file = configs.input_files_and_folders.mesh_file.cdata.strip()

    e_ref = np.array(list(map(float, configs.physical_variables.e_ref.cdata.split(','))))
    K1_form = np.array(list(map(float, configs.physical_variables.K1_form.cdata.split(',')))).reshape((3, 3))

    result_folder = configs.output_files_and_folders.res_fldr.cdata.strip()
    save_subres = 'True' == configs.output_files_and_folders.save_subres.cdata.strip()

    return mesh_file, e_ref, K1_form, result_folder, save_subres


# TODO: never used ?
def permeability_initialiser_config_reader_yml(input_file_path):
    """
    Load and parse a permeability initialisation configuration from a YAML file.

    This function opens a YAML configuration file, loads its content, and converts
    the physical configuration fields `K1_form` and `e_ref` into NumPy arrays
    for further use in computations.

    Args:
        input_file_path (str): Path to the YAML configuration file.

    Returns:
        configs (dict): A dictionary containing all configuration parameters,
                        with 'physical.K1_form' reshaped to (3, 3) NumPy array
                        and 'physical.e_ref' as a NumPy array of shape (3,).
    """
    with open(input_file_path, "r") as config_file:
        configs = yaml.load(config_file, yaml.SafeLoader)

    configs['physical']['K1_form'] = np.array(configs['physical']['K1_form']).reshape((3, 3))
    configs['physical']['e_ref'] = np.array(configs['physical']['e_ref'])

    return configs


def initialise_permeabilities(K1_space, K2_space, permeability_folder, **kwarg):
    """
    Initialise permeability tensor fields from XDMF files for different model types.

    This function reads the K1 permeability tensor from a stored XDMF file located in
    `permeability_folder`. It also prepares K2 (empty for now) and K3 as a deep copy
    of K1. The permeability fields are set depending on the chosen model type.

    Args:
        K1_space (FunctionSpace): The function space for the K1 tensor field.
        K2_space (FunctionSpace): The function space for the K2 tensor field.
        permeability_folder (str): Path to the folder containing permeability XDMF files.
        **kwarg:
            model_type (str): The permeability model to use. Can be:
                              - 'acv': arteries-capillaries-veins (default)
                              - 'a': arteries only

    Returns:
        tuple:
            K1 (Function): The initialised K1 permeability tensor.
            K2 (Function): The initialised (empty) K2 permeability tensor.
            K3 (Function): A deep copy of K1.

    Raises:
        Exception: If `model_type` is unknown.
    """
    model_type = kwarg.get('model_type', 'acv')

    comm = MPI.comm_world

    # TODO: determine why there is two versions but it is the same
    if model_type == 'acv':
        K1 = Function(K1_space)
        K2 = Function(K2_space)

        with XDMFFile(comm, permeability_folder + "K1_form.xdmf") as file:
            file.read_checkpoint(K1, "K1_form")
        K3 = K1.copy(deepcopy=True)
    elif model_type == 'a':
        K1 = Function(K1_space)
        K2 = Function(K2_space)

        with XDMFFile(comm, permeability_folder + "K1_form.xdmf") as file:
            file.read_checkpoint(K1, "K1_form")
        K3 = K1.copy(deepcopy=True)
    else:
        raise Exception("Unknown model type: " + model_type)

    return K1, K2, K3