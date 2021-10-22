"""
Generate mesh for verification

@author: Tamás Józsa
"""


#%% mesh file definition
class MSHfile:
    """
    Class to read and write msh mesh files.
    """

    def __init__(self):
        self.MeshFormat = []
        self.PhysicalNames = []
        self.Nodes = []
        self.Elements = []

    def Loadfile(self, file):
        """
        Load the msh file into the object.
        :param file: msh file
        :return: Nothing
        """
        print("Loading MSH: %s" % file)
        mesh_raw = [i.strip('\n') for i in open(file)]
        mesh = [i.split(' ') for i in mesh_raw]

        startelementsFormat = mesh_raw.index("$MeshFormat")
        endelementsFormat = mesh_raw.index("$EndMeshFormat")
        startelementsNames = mesh_raw.index("$PhysicalNames")
        endelementsNames = mesh_raw.index("$EndPhysicalNames")
        startelementsNodes = mesh_raw.index("$Nodes")
        endelementsNodes = mesh_raw.index("$EndNodes")
        startelements = mesh_raw.index("$Elements")
        endelements = mesh_raw.index("$EndElements")

        self.MeshFormat = [mesh_raw[i] for i in range(startelementsFormat + 1, endelementsFormat)]
        self.PhysicalNames = [[int(mesh[i][0]), int(mesh[i][1]), mesh[i][2]] for i in
                              range(startelementsNames + 2, endelementsNames)]
        self.Nodes = [[int(mesh[i][0]), float(mesh[i][1]), float(mesh[i][2]), float(mesh[i][3])] for i
                      in range(startelementsNodes + 2, endelementsNodes)]
        self.Elements = [[int(x) for x in mesh[i] if x] for i in range(startelements + 2, endelements)]

    def Writefile(self, file):
        """
        Write the object to file
        :param file: file name
        :return: Nothing
        """
        print("Writing MSH: %s" % file)
        with open(file, 'w') as f:
            f.write("$MeshFormat\n")
            for i in self.MeshFormat:
                f.write(i + "\n")
            f.write("$EndMeshFormat\n")

            f.write("$PhysicalNames\n")
            f.write(str(len(self.PhysicalNames)) + "\n")
            for i in self.PhysicalNames:
                line = ' '.join(str(x) for x in i)
                f.write(line + "\n")
            f.write("$EndPhysicalNames\n")

            f.write("$Nodes\n")
            f.write(str(len(self.Nodes)) + "\n")
            for i in self.Nodes:
                line = ' '.join(str(x) for x in i)
                f.write(line + "\n")
            f.write("$EndNodes\n")

            f.write("$Elements\n")
            f.write(str(len(self.Elements)) + "\n")
            for i in self.Elements:
                line = ' '.join(str(x) for x in i)
                f.write(line + "\n")
            f.write("$EndElements\n")

    def GetElements(self, ids):
        """
        Get the index and elements of the mesh at specific regions.
        :param ids: regions ids
        :return: elements, indexes
        """
        data = [[index, element] for index, element in enumerate(self.Elements) if
                int(element[4]) in ids or int(element[3]) in ids]
        indexes = [i[0] for i in data]
        elements = [i[1] for i in data]
        return elements, indexes

    def GetSurfaceCentroids(self, ids):
        """
        Get the centroids of the elements of the mesh
        :param ids: region ids of the elements
        :return: Positions, elements, indexes
        """
        elements, indexes = self.GetElements(ids)

        triangles = [[i[-1], i[-2], i[-3]] for i in elements]

        trianglespos = [[self.Nodes[triangle[0] - 1][1:],
                         self.Nodes[triangle[1] - 1][1:],
                         self.Nodes[triangle[2] - 1][1:]
                         ] for triangle in triangles]

        positions = [meanpos(i) for i in trianglespos]
        return positions, elements, indexes

    def AreaRegion(self, regionid):
        """
        Get the area of the elements of some region.
        :param regionid: region id of the elements
        :return: total area, number of triangles.
        """
        elements, indexes = self.GetElements([regionid])
        triangles = [[i[-1], i[-2], i[-3]] for i in elements]
        trianglespos = [[self.Nodes[triangle[0] - 1][1:],
                         self.Nodes[triangle[1] - 1][1:],
                         self.Nodes[triangle[2] - 1][1:]
                         ] for triangle in triangles]
        areas = [TriangleToArea(triangle) for triangle in trianglespos]
        totalarea = sum(areas)
        return totalarea, len(triangles)

def TriangleToArea(nodes):
    """
    Take a list of node positions and return the area using Heron's formula.
    :param nodes: Set of points
    :return: area
    """
    a = distancebetweenpoints(nodes[0], nodes[1])
    b = distancebetweenpoints(nodes[1], nodes[2])
    c = distancebetweenpoints(nodes[0], nodes[2])
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def meanpos(nodes):
    """
    Take a list of node positions and return the centroid (mean position).
    :param nodes: list of node positions
    :return: mean x,y,z coordinates
    """
    x = np.mean([nodes[0][0], nodes[1][0], nodes[2][0]])
    y = np.mean([nodes[0][1], nodes[1][1], nodes[2][1]])
    z = np.mean([nodes[0][2], nodes[1][2], nodes[2][2]])
    return [x, y, z]


def distancebetweenpoints(p1, p2):
    """
    Calculate the euclidean distance between two points.
    :param p1: List of coordinates
    :param p2: List of coordinates
    :return:
    """
    dsq = sum([np.square(p1[i] - p2[i]) for i in range(0, len(p1))])
    return np.sqrt(dsq)


#%%
import os
import numpy as np
import dolfin
import meshio

import yaml
import analyt_fcts
import sys
sys.path.append('../')
import convert_msh2hdf5
import IO_fcts

with open('config_coupled_analyt.yaml', "r") as configfile:
        configs = yaml.load(configfile, yaml.SafeLoader)
D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw = analyt_fcts.set_up_network(configs)
beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con = analyt_fcts.set_up_continuum(configs)

Lx = np.sum(l_subdom)*1000
nx = configs['numerical']['nx']

Ly = np.sqrt(configs['continuum']['area'])*1000
Lz = np.sqrt(configs['continuum']['area'])*1000

p1 = dolfin.Point(0,0,0)
p2 = dolfin.Point(Lx,Ly,Lz)

mesh = dolfin.BoxMesh(p1, p2, nx, 1, 1)

vertices = mesh.coordinates()
elements = mesh.cells()


#%% label elements
subdomain_limits = []
for i in range(len(l_subdom)):
    if i==0:
        subdomain_limits.append( [0,l_subdom[0]*1000] )
    else:
        subdomain_limits.append( [subdomain_limits[i-1][1], subdomain_limits[i-1][1]+1000*l_subdom[i]] )

# half GM and half WM
element_centr_coord = np.zeros([elements.shape[0],3])
element_label = np.zeros(elements.shape[0],dtype=np.int64)
for i in range(elements.shape[0]):
    element_centr_coord[i,:] = np.mean(vertices[elements[i],:],axis=0)
    for j in range(len(l_subdom)):
        if element_centr_coord[i,0]>subdomain_limits[j][0] and element_centr_coord[i,0]<=subdomain_limits[j][1]:
            element_label[i] = configs['numerical']['layers'][j]

#%% obtain boundary surface
facets = np.concatenate((elements[:,[0,1,2]],elements[:,[0,1,3]],elements[:,[0,2,3]],elements[:,[1,2,3]]),axis=0)
facets = np.sort(facets,1)
facets = facets[ np.lexsort(np.fliplr(facets).T) ]

diffs = np.all(np.diff(facets,axis=0) == 0, axis=1)
duploc = np.where(diffs)
# facets = np.delete(facets,duploc[0],axis=0)
duplocs = np.concatenate((duploc[0],duploc[0]+1),axis=0)
facets = np.delete(facets,duplocs,axis=0)


#%% label boundary surface
facet_centr_coord = np.zeros(facets.shape)
facet_label = np.zeros(facets.shape[0],dtype=np.int64)
for i in range(facets.shape[0]):
    facet_centr_coord[i,:] = np.mean(vertices[facets[i],:],axis=0)
    if dolfin.near(facet_centr_coord[i,0],0):
        facet_label[i] = 20
    elif dolfin.near(facet_centr_coord[i,0],Lx):
        facet_label[i] = 21
    elif dolfin.near(facet_centr_coord[i,1],0) or dolfin.near(facet_centr_coord[i,1],Ly):
        facet_label[i] = 1
    elif dolfin.near(facet_centr_coord[i,2],0) or dolfin.near(facet_centr_coord[i,2],Lz):
        facet_label[i] = 2


#%% save boundary surface

if not os.path.exists('verification_mesh'):
    os.makedirs('verification_mesh')
os.chdir('./verification_mesh')

cells = [
    ("triangle", facets)
]

surf_mesh = meshio.Mesh(vertices,cells,
                        cell_data={"BC_ID": [facet_label]}
                        )

meshio.write(
    'boundary_mesh.vtk',surf_mesh,
    )


#%% generate msh file
msh_nodes = []
for i in range(vertices.shape[0]):
    msh_nodes.append( [i+1,vertices[i,0],vertices[i,1],vertices[i,2]] )

facets = np.column_stack(   (np.arange(1,facets.shape[0]+1),
                             2*np.ones(facets.shape[0],dtype=np.int64),
                             2*np.ones(facets.shape[0],dtype=np.int64),
                             facet_label,
                             facet_label,
                             facets+1) )
elements = np.column_stack( (np.arange(facets.shape[0],facets.shape[0]+elements.shape[0]),
                             4*np.ones(elements.shape[0],dtype=np.int64),
                             2*np.ones(elements.shape[0],dtype=np.int64),
                             element_label,
                             element_label,
                             elements+1) )

msh_elements = []
for i in range(facets.shape[0]):
    msh_elements.append( list(facets[i,:]) )
for i in range(elements.shape[0]):
    msh_elements.append( list(elements[i,:]) )

physical_names = [[2,1,'"yBC"'],[2,2,'"zBC"'],[2,20,'"xBC1"'],[2,21,'"xBC2"'],[3,11,'"white_matter"']]

mymesh = MSHfile()
mymesh.Elements = msh_elements
mymesh.Nodes = msh_nodes
mymesh.PhysicalNames = physical_names
mymesh.MeshFormat = ['2.2 0 8']
mymesh.Writefile('labelled_box_mesh.msh')

convert_msh2hdf5.convert_mesh('labelled_box_mesh.msh', 'labelled_box_mesh')


#%% generate permeability form
mesh, subdomains, boundaries = IO_fcts.mesh_reader('labelled_box_mesh.xdmf')
K_space = dolfin.TensorFunctionSpace(mesh, "DG", 0)
K1 = dolfin.Function(K_space)
K1_array = K1.vector()[:]
K1_loc = np.reshape( np.eye(3),9 )
for i in range(mesh.num_cells()):
    K1_array[i*9:i*9+9] = K1_loc
K1.vector()[:] = K1_array
with dolfin.XDMFFile('K1_form.xdmf') as myfile:
    myfile.write_checkpoint(K1,"K1_form", 0, dolfin.XDMFFile.Encoding.HDF5, False)
