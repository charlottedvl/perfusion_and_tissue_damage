# Python imports
import yaml
from dolfinx import fem
import numpy as np

# Local imports
from ..io import IO_functions


def permeability_initialiser_config_reader_yaml(input_file_path: str) -> dict:
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

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
        KeyError: If required fields are missing.
        ValueError: If reshaping or array conversion fails.
    """
    try:
        with open(input_file_path, "r") as config_file:
            configs = yaml.load(config_file, yaml.SafeLoader)

        configs['physical']['K1_form'] = np.array(configs['physical']['K1_form']).reshape((3, 3))
        configs['physical']['e_ref'] = np.array(configs['physical']['e_ref'])

    except FileNotFoundError:
        IO_functions.print0(f"Error: Configuration file not found at '{input_file_path}'")
    except yaml.YAMLError as e:
        IO_functions.print0("Error parsing YAML file:", e)
    except KeyError as e:
        IO_functions.print0(f"Missing required configuration key: {e}")
    except ValueError as e:
        IO_functions.print0("Error converting configuration values to NumPy arrays:", e)
    except Exception as e:
        IO_functions.print0("An unexpected error occurred while reading the configuration file:", e)

    return configs


def initialise_permeabilities(K1_space, K2_space, permeability_folder, **kwarg):
    """
    Initialise the three permeability tensor components K1, K2, and K3.

    1. Reads a model type flag from the input arguments (default: 'acv').
    2. Allocates K1 and K2 as fem.Functions in the provided function spaces.
    3. Loads K1 from file 'K1_form.h5' inside the specified permeability folder.
    4. Copies the values of K1 into K3 (which shares K1's function space).
    5. If the model type is not recognised, an exception is raised.

    Args:
        K1_space (dolfinx.fem.FunctionSpace): Function space for K1.
        K2_space (dolfinx.fem.FunctionSpace): Function space for K2.
        permeability_folder (str): Directory path containing "K1_form.h5".
        **kwarg:
            model_type (str): Either 'acv' or 'a'. Default 'acv'.

    Returns:
        tuple:
            K1 (dolfinx.fem.Function): Arterial permeability tensor read from file.
            K2 (dolfinx.fem.Function): Capillary permeability (allocated, not read from file).
            K3 (dolfinx.fem.Function): Venous permeability — deep copy of K1.

    Raises:
        Exception: If model_type is not 'acv' or 'a'.
    """
    # Step 1: Determine model type
    model_type = kwarg.get('model_type', 'acv')

    if model_type is not 'acv' and model_type is not 'a':
        IO_functions.print0(f"ERROR: Unknown model type '{model_type}' encountered.")
        raise Exception("unknown model type: " + model_type)

    # Step 2: Allocate functions
    K1 = fem.Function(K1_space)
    K2 = fem.Function(K2_space)

    # Step 3: Load K1 from file
    filename = permeability_folder + "K1_form.h5"
    IO_functions.read_function_from_h5(K1, filename, "Function")

    # Step 4: Create K3 as a deep copy of K1
    K3 = fem.Function(K1.function_space)
    K3.x.array[:] = K1.x.array.copy()

    K3.x.scatter_forward()

    return K1, K2, K3