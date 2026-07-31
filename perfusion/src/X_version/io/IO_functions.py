# Python imports
import h5py
from mpi4py import MPI
import numpy as np
import os
from typing import Dict, Any, Optional
import yaml
import argparse
from scipy.interpolate import interp1d

# Dolfin-x imports
import basix.ufl
import dolfinx
import dolfinx.io as io
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh, MeshTags

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# ----------------------------------------Reading and Processing files----------------------------------------
def print0(*args: Any, **kwargs: Any):
    """Print arguments only if rank is 0.

    This function acts as a conditional wrapper around print, typically used
    in parallel computing to restrict output to a single process (rank 0).
    It accepts arguments and keyword arguments (key-value pairs).

    Parameters:
        *args : Any
            - accepts any number of positional arguments of any data type.
        **kwargs : Any
            - 'keyword arguments' - those passed as 'key:value'.
    """

    if rank == 0:
        print(*args, **kwargs)


def mesh_reader(comm, mesh_file):
    """
    Loads a volumetric mesh and associated subdomain and boundary information from XDMF files.

    This function reads a mesh, its physical region tags (volume subdomains), and its facet tags
    (boundary surfaces) from separate XDMF files, as per the provided XDMF structure.

    Args:
        comm (mpi4py.MPI.Comm): The MPI communicator.
        mesh_file (str): Path to the main volumetric mesh file (e.g., './brain_meshes/b0000/clustered.xdmf').

    Returns:
        tuple: A tuple containing:
            - mesh (dolfinx.mesh.Mesh): The loaded mesh object.
            - subdomains (dolfinx.mesh.MeshTags or None): Subdomain markers (dimension 3, cell tags).
            - boundaries (dolfinx.mesh.MeshTags or None): Boundary markers (dimension 2, facet tags).
    """
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    base_file_name = os.path.splitext(mesh_file)[0] # suppress extension

    subdomain_file = base_file_name + '_physical_region.xdmf'
    boundary_file = base_file_name + '_facet_region.xdmf'

    # --- Read the main mesh ---
    mesh = None
    with io.XDMFFile(comm, mesh_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # --- Read subdomains (cell tags) ---
    subdomains = None
    if os.path.exists(subdomain_file):
        with io.XDMFFile(comm, subdomain_file, "r") as xdmf_sub:
            subdomains = xdmf_sub.read_meshtags(mesh, name="mesh")
    else:
        print0(f"Subdomain file not found: {subdomain_file}. Subdomains will be None.")

    # --- Read boundaries (facet tags) ---
    boundaries = None
    if os.path.exists(boundary_file):
        with io.XDMFFile(comm, boundary_file, "r") as xdmf_bnd:
            boundaries = xdmf_bnd.read_meshtags(mesh, name="mesh")
    else:
        print0(f"Boundary file not found: {boundary_file}. Boundaries will be None.")

    return mesh, subdomains, boundaries


def read_function_from_h5(K, filename: str, dataset_name="Function"):
    """
    Read global (owned) data from HDF5 and scatter into the DolfinX Function `K`.

    Supports two HDF5 layouts:
    - FEniCS-X format:  ``Function/f/0``  (written by XDMFFile.write_function)
    - Legacy format:    ``<name>/<name>_0/vector``  (written by XDMFFile.write_checkpoint)

    The function auto-detects the layout by looking for a ``vector`` dataset
    inside the group, falling back to the generic key traversal otherwise.

    Parameters
    ----------
    K : dolfinx.fem.Function
        Target function to populate with global (non-ghost) values.
    filename : str
        Path to the HDF5 file storing the global DoF values.
    dataset_name : str
        Top-level group name, e.g. ``"K1_form"`` or ``"Function"``.
    """
    import h5py
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # --- Step 1: Read the values dataset ---
    flat_data = None
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:

        # Priority 1: Legacy write_checkpoint layout → <name>/<name>_0/vector
        legacy_key = f"{dataset_name or 'Function'}/{dataset_name or 'Function'}_0/vector"
        # Priority 2: generic traversal (handles FEniCS-X Function/f/0 layout)
        generic_key = find_dataset_key(f, dataset_name or "Function")

        if legacy_key in f and isinstance(f[legacy_key], h5py.Dataset):
            dataset_key = legacy_key
        elif generic_key is not None and isinstance(f[generic_key], h5py.Dataset):
            dataset_key = generic_key
        else:
            raise KeyError(
                f"[Rank {rank}] Could not find values dataset for '{dataset_name}' in '{filename}'. "
                f"Available top-level keys: {list(f.keys())}")

        raw_data = f[dataset_key][()]

        if raw_data.ndim == 2:
            flat_data = raw_data.reshape(-1)
        elif raw_data.ndim == 1:
            flat_data = raw_data
        else:
            raise ValueError(f"[Rank {rank}] Unsupported data shape: {raw_data.shape}")

    # --- Step 2: Determine ownership (exclude ghost cells) ---
    index_map = K.function_space.dofmap.index_map
    bs = K.function_space.dofmap.index_map_bs  # block size (9 for tensor, 1 for scalar)
    local_size = index_map.size_local * bs      # number of owned values in x.array

    # Compute offset in global array
    offset = comm.scan(local_size) - local_size
    start, end = int(offset), int(offset + local_size)

    # --- Step 3: Assign data to owned DoFs only ---
    K.x.array[:local_size] = flat_data[start:end]
    K.x.scatter_forward()
    return K.x.array


def xdmf_reader(variable, variable_name, folder):
    file_path = folder + variable_name + '.xdmf'
    with io.XDMFFile(MPI.COMM_WORLD, file_path, "r") as myfile:
        if isinstance(variable, dolfinx.mesh.Mesh):
            myfile.read_mesh(variable)
            myfile.read_mesh(variable, variable_name)
    return variable


def hdf5_reader(mesh, variable, variable_name, folder):
    file_path = folder + variable_name + '.h5'

    with h5py.File(file_path, "r") as hdf:
        print(f"Reading {variable_name} from {file_path}")

        # Read data from the HDF5 file
        if variable_name not in hdf:
            raise KeyError(f"Variable '{variable_name}' not found in HDF5 file: {file_path}")

        if isinstance(hdf[variable_name], h5py.Group):
            if "vector_0" in hdf[variable_name]:
                dataset = hdf[variable_name]["vector_0"]
            else:
                raise KeyError(f"No 'vector_0' dataset found in group '{variable_name}'")
        else:
            dataset = hdf[variable_name]

        data = np.array(dataset)

    if data.shape[0] != variable.x.array.shape[0]:
        print("Resizing HDF5 data to match FEniCSx variable size.")
    old_indices = np.linspace(0, 1, data.shape[0])
    new_indices = np.linspace(0, 1, variable.x.array.shape[0])
    interpolator = interp1d(old_indices, data, kind='linear', fill_value='extrapolate')
    data = interpolator(new_indices)

    # Assign data to the FEniCSx function variable
    variable.x.array[:] = data

    return variable


def find_dataset_key(f, key):
    """Recursively find the correct dataset key."""
    import h5py
    from mpi4py import MPI

    while key in f and isinstance(f[key], h5py.Group):
        sub_keys = list(f[key].keys())
        if not sub_keys:
            raise KeyError(f"Group '{key}' is empty.")
        key += f"/{sub_keys[0]}"
    if key not in f:
        raise KeyError(f"Key '{key}' not found. Available keys: {list(f.keys())}")
    return key


# %%
# ----------------------------------------Reading Configurations----------------------------------------



class dict2obj(dict):
    def __init__(self, my_dict):
        for a, b in my_dict.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)