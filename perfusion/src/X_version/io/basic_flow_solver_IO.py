import os
import dolfinx
from mpi4py import MPI
import yaml


def basic_flow_config_reader_yaml(input_file_path, parser):
    """
    Load and update simulation configuration from a YAML file and command-line parser overrides.

    This function reads simulation parameters from a `.yaml` configuration file and overrides
    specific fields (if provided) using arguments parsed from an `argparse.ArgumentParser` instance.
    It also ensures the results folder exists and saves the final configuration into it.

    Args:
        input_file_path (str): Path to the YAML configuration file.
        parser (argparse.ArgumentParser): Argument parser containing optional overrides.

    Returns:
        dict: Parsed and possibly updated configuration dictionary.

    Raises:
        Exception: If the input file format is unsupported (non-YAML).
    """
    if input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as config_file:
            configs = yaml.load(config_file, yaml.SafeLoader)
    else:
        config_format = os.path.splitext(input_file_path)[-1]
        raise Exception("Unknown input file format: " + config_format)

    if hasattr(parser.parse_args(), 'res_fldr'):
        if parser.parse_args().res_fldr is not None:
            configs['output']['res_fldr'] = parser.parse_args().res_fldr

    if hasattr(parser.parse_args(), 'mesh_file'):
        if parser.parse_args().mesh_file is not None:
            configs['input']['mesh_file'] = parser.parse_args().mesh_file

    if hasattr(parser.parse_args(), 'inlet_boundary_file'):
        if parser.parse_args().inlet_boundary_file is not None:
            configs['input']['inlet_boundary_file'] = parser.parse_args().inlet_boundary_file

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        if not os.path.exists(configs['output']['res_fldr']):
            os.makedirs(configs['output']['res_fldr'])
        with open(configs['output']['res_fldr'] + 'settings.yaml', 'w') as outfile:
            yaml.dump(configs, outfile, default_flow_style=False)

    return configs


class dict2obj(dict):
    def __init__(self, my_dict):
        for a, b in my_dict.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)