def extract_files_from_config(args, result_folder):
    """
    Determine the file paths for baseline and occluded perfusion data.

    Uses values from the parsed command-line arguments if provided.
    If not, defaults to standard filenames within the specified result folder.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing optional paths.
        result_folder (str): Path to the results folder used for default file resolution.

    Returns:
        tuple[str, str]: A tuple containing:
            - healthy_file (str): Path to baseline (healthy) perfusion XDMF file.
            - occluded_file (str): Path to occluded (stroke) perfusion XDMF file.
    """
    if args.baseline is None:
        healthy_file = result_folder + 'perfusion.xdmf'
    else:
        healthy_file = args.baseline
    if args.occluded is None:
        occluded_file = getattr(args, 'occluded', result_folder + 'perfusion_stroke.xdmf')
    else:
        occluded_file = args.occluded
    return healthy_file, occluded_file


def prepare_compartmental_model(simulation_configs):
    """
    Extract the compartmental model type from simulation configuration.

    Defaults to 'acv' if the model type is not specified.

    Args:
        simulation_configs (dict): Simulation parameters dictionary.

    Returns:
        str: Model type string in lowercase.
    """
    try:
        return simulation_configs.get('model_type').lower().strip()
    except KeyError:
        return 'acv'


def prepare_velocity_order(simulation_configs):
    """
    Determine the velocity order for finite element approximation.

    Uses 'vel_order' if present in the configuration; otherwise,
    returns one degree less than 'fe_degr'.

    Args:
        simulation_configs (dict): Simulation parameters dictionary.

    Returns:
        int: Velocity order for finite elements.
    """
    try:
        return simulation_configs.get('vel_order')
    except KeyError:
        return simulation_configs.get('fe_degr') - 1


def prepare_simulation_parameters(simulation_configs):
    """
    Prepare simulation parameters from the configuration dictionary.

    Extracts and returns the compartmental model type and velocity order
    based on the provided simulation configuration.

    Args:
        simulation_configs (dict): Simulation parameters dictionary.

    Returns:
        tuple: (compartmental_model (str), velocity_order (int or None))
    """
    compartmental_model = prepare_compartmental_model(simulation_configs)
    velocity_order = prepare_velocity_order(simulation_configs)
    return compartmental_model, velocity_order