# HPC Submission Scripts

This folder contains job submission scripts for running simulations on HPC clusters.

Two supercomputers are supported by this repository: 
- Ares, a Polish supercomputer. Find more about Ares [here](https://www.cyfronet.pl/en/computers/18827,artykul,ares_supercomputer.html);
- Crescent2, [Cranfield University](https://www.cranfield.ac.uk/) supercomputer. Find more about Crescent2 [here](https://www.cranfield.ac.uk/academic-disciplines/computing-simulation-and-modelling).

Scripts typically:
- Load required modules (e.g., FEniCS, MPI)
- Set job resource limits (e.g., nodes, time, memory)
- Run the Python script with a specific config
