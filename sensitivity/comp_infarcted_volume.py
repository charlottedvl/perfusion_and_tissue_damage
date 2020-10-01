from dolfin import *
import numpy as np
import IO_fcts

case1_folder = './healthy00/'
case2_folder = './RMCA_occlusion00/'

mesh_file = '../brain_meshes/b0000/clustered.xdmf'

mesh, subdomains, boundaries = IO_fcts.mesh_reader(mesh_file)

V = FunctionSpace(mesh, "DG", 0)

Perf1 = Function(V)
Perf2 = Function(V)

with XDMFFile(comm,mesh_file[:-5]+'_facet_region.xdmf') as myfile: myfile.read(boundaries)

IO_fcts.hdf5_reader( mesh,Perf1,case1_folder+'results/','perfusion.h5','P' )
IO_fcts.hdf5_reader( mesh,Perf2,case2_folder+'results/','perfusion.h5','P' )

perf_change = project(100*(Perf2-Perf1)/Perf1, V, solver_type='bicgstab', preconditioner_type='amg')

mask = perf_change.vector()[:]<-70
cell_indices = np.argwhere(mask==True)

vertex_IDs = mesh.cells()[cell_indices[:,0]]
vertex_coordinates = []
volumes = []

for i in range(len(vertex_IDs)):
    vertex_coordinates.append(mesh.coordinates()[vertex_IDs[i]])
    vec1 = vertex_coordinates[i][0]-vertex_coordinates[i][3]
    vec2 = vertex_coordinates[i][1]-vertex_coordinates[i][3]
    vec3 = vertex_coordinates[i][2]-vertex_coordinates[i][3]
    volumes.append( abs(np.dot(vec1,np.cross(vec2,vec3)))/6  )

volumes = np.array(volumes)
V_brain = assemble(Constant(1.0)*dx(domain=mesh))

print( 100*sum(volumes)/V_brain )