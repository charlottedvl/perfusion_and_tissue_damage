# perfusion_and_tissue_damage

This repository contains software corresponding to WP5 of the INSIST project (https://www.insist-h2020.eu/).

The folders have the following purpose:

1; brain_meshes.tar.xz: archive of patient specific labelled tetrahedral meshes sufficient for finite element simulations.

2; perfusion: multi-compartment Darcy flow solver describin blood flow in healthy and occluded scenarios.

3; oxygen: multi-compartment advection-reaction-diffusion problem describing oxygen transport in the brain.

4; tissue_health: metabolism model capturing tissue damage as a function of time before, during, and after stroke treatment.


**Required inputs from other modules:**

1; Patient parameters (YAML): age, sex, systolic and diastolic blood pressure, heart rate, occlusion location <-- WP2

2; Treatment outcome: thrombus permeability before and after treatment, distribution of thrombus fragments if any organised in a file listing diameter and number of fragments (*.csv prefered) <-- WP3 & WP4

3; Size and location of infarct as a function of treatment success (TICI), collateral score, age and sex for validation (*.csv prefered) <-- WP2



**Provided outputs for other modules:**

1; Pressure upstream and downstream of thrombus (*.csv) --> WP3

2; Patient outcome (size and location of infarct, TICI, functional outcome estimate) as a function of patient parameters (*.csv) --> WP7