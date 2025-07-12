# Simulation folder

This folder contains all simulation scripts. Three main are presents:
- BC_creator.py;
- permeability_initialiser.py;
- basic_flow_solver.py. 

Please add all physics-driven and numerically intensive simulation here. A small review of the simulation is present below. 

## BC_creator.py

The [BC_creator.py](BC_creator.py) is responsible for the creation of boundary data. The file created is saved in the 
[boundary_data folder](../../../boundary_data). 

## permeability_initialiser.py

The [permeability_initialiser.py](permeability_initialiser.py) generates an anisotropic permeability field based on a 
predefined form of the permeability tensor given in a reference coordinate system. 
The output is saved inside the [brain_meshes folder, in a permeability folder](../../../brain_meshes/b0000). The location can be updated with
the configuration file. 

## basic_flow_solver.py

The [basic_flow_solver.py](basic_flow_solver.py) is a multi-compartment Darcy flow model with mixed Dirichlet and 
Neumann boundary conditions. 
The output is saved inside the [results folder](../../../results). The location can be updated with the configuration file. 