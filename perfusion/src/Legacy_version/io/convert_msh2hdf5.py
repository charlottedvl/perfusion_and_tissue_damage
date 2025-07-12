import dolfin
import os
import sys


def convert_mesh_to_xdmf(input_mesh_file, xdmf_mesh_file):
    """
    Converts a .msh file to .xdmf format for use with FEniCS, extracting subdomains and boundaries.

    Args:
       input_mesh_file (str): Path to the input .msh mesh file.
       xdmf_mesh_file (str): Base name (no extension) for the output .xdmf files.

    Returns:
       int: 0 if successful.
    """
    # Step 1: Convert .msh to .xml using dolfin-convert
    cmd = 'dolfin-convert ' + input_mesh_file +' ' +  input_mesh_file[0:-3] + 'xml'
    os.system(cmd)

    # Step 2: Load mesh and region data
    mesh = dolfin.Mesh(input_mesh_file[0:-3] + 'xml')
    subdomains = dolfin.MeshFunction("size_t", mesh, input_mesh_file[0:-4] + '_physical_region.xml')
    boundaries = dolfin.MeshFunction("size_t", mesh, input_mesh_file[0:-4] + '_facet_region.xml')

    # Step 3: Save mesh and labels in XDMF format
    xdmf_msh_file = dolfin.XDMFFile(xdmf_mesh_file + '.xdmf')
    xdmf_msh_file.write(mesh)
    xdmf_subdom_file = dolfin.XDMFFile(xdmf_mesh_file  + '_physical_region.xdmf')
    xdmf_subdom_file.write(subdomains)
    xdmf_boundaries_file = dolfin.XDMFFile(xdmf_mesh_file  + '_facet_region.xdmf')
    xdmf_boundaries_file.write(boundaries)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_mesh_file = 'clustered_mesh.msh'
        xdmf_mesh_file = 'clustered_mesh'

    if len(sys.argv) == 3:
        input_mesh_file = sys.argv[1]
        xdmf_mesh_file = sys.argv[2]

    print("Converting {} to {}.".format(input_mesh_file, xdmf_mesh_file))
    sys.exit(convert_mesh_to_xdmf(input_mesh_file, xdmf_mesh_file))
