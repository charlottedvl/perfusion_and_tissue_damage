import dolfin
import os
import sys


def convert_mesh(labelled_mesh_file, new_mesh_name):
    cmd = 'dolfin-convert ' + labelled_mesh_file +' ' +  labelled_mesh_file[0:-3] + 'xml'
    os.system(cmd)

    mesh = dolfin.Mesh(labelled_mesh_file[0:-3] + 'xml')
    subdomains = dolfin.MeshFunction("size_t", mesh, labelled_mesh_file[0:-4] + '_physical_region.xml')
    boundaries = dolfin.MeshFunction("size_t", mesh, labelled_mesh_file[0:-4] + '_facet_region.xml')

    # new implementation
    xdmf_msh_file = dolfin.XDMFFile(new_mesh_name + '.xdmf')
    xdmf_msh_file.write(mesh)
    xdmf_subdom_file = dolfin.XDMFFile(new_mesh_name  + '_physical_region.xdmf')
    xdmf_subdom_file.write(subdomains)
    xdmf_boundaries_file = dolfin.XDMFFile(new_mesh_name  + '_facet_region.xdmf')
    xdmf_boundaries_file.write(boundaries)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        labelled_mesh_file = 'clustered_mesh.msh'
        new_mesh_name = 'clustered_mesh'

    if len(sys.argv) == 3:
        labelled_mesh_file = sys.argv[1]
        new_mesh_name = sys.argv[2]

    print("Converting {} to {}.".format(labelled_mesh_file, new_mesh_name))
    sys.exit(convert_mesh(labelled_mesh_file, new_mesh_name))
