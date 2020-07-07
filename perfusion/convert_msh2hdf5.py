import numpy
import dolfin
import os
import sys

labelled_mesh_file = 'clustered_mesh.msh'
new_mesh_name = 'clustered_mesh'


cmd = 'dolfin-convert ' + labelled_mesh_file +' ' +  labelled_mesh_file[0:-3] + 'xml'
os.system(cmd)


mesh = dolfin.Mesh(labelled_mesh_file[0:-3] + 'xml')                                                  
subdomains = dolfin.MeshFunction("size_t", mesh, labelled_mesh_file[0:-4] + '_physical_region.xml')    
boundaries = dolfin.MeshFunction("size_t", mesh, labelled_mesh_file[0:-4] + '_facet_region.xml')

# # old implementation
# hdf = dolfin.HDF5File(mesh.mpi_comm(), labelled_mesh_file[0:-3] + 'h5', "w")
# hdf.write(mesh, "/mesh")
# hdf.write(subdomains, "/subdomains")
# hdf.write(boundaries, "/boundaries")
# hdf.close()

# new implementation
xdmf_msh_file = dolfin.XDMFFile(new_mesh_name + '.xdmf')
xdmf_msh_file.write(mesh)
xdmf_subdom_file = dolfin.XDMFFile(new_mesh_name  + '_physical_region.xdmf')
xdmf_subdom_file.write(subdomains)
xdmf_boundaries_file = dolfin.XDMFFile(new_mesh_name  + '_facet_region.xdmf')
xdmf_boundaries_file.write(boundaries)
