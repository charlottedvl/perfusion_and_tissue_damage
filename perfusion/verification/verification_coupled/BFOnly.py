#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""1D Blood Flow generator

Usage:
  BF.py <patient_folder>
  BF.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from datetime import datetime

from Blood_Flow_1D import Patient, docopt, transcript


def generatebloodflowfiles(patient_folder):
    """
    Generate blood flow files for a simulation. This creates the minimally required files to run a blood flow simulation.
    Useful for running simple networks.

    Parameters
    ----------
    patient_folder : str
        folder with input files
    """
    # Load input files
    patient = Patient.Patient(patient_folder)
    patient.ResetModellingFolder()

    patient.Load1DAnatomy("1-D_Anatomy.txt")
    patient.LoadPatientData()
    patient.LoadModelParameters()
    patient.UpdateModelParameters()
    patient.ModelParameters["coarse_collaterals_number"] = int(0)

    patient.GenerateTrees(8)  # cut-off radius based on Murray's law
    patient.AddTreesToTop()
    for node in patient.Topology.Nodes:
        node.YoungsModules = 1e6
    for vessel in patient.Topology.Vessels:
        vessel.YoungsModules = 1e6

    patient.calculate_wk_parameters_evenly()
    patient.ImportClots()
    patient.WriteModelParameters()
    patient.WriteSimFiles()
    patient.TopologyToVTP()

    with open(patient.Folders.ModellingFolder + "Clusters.csv", 'w') as f:
        f.write("NodeID,Position,Number of CouplingPoints,ClusterID,Area,MajorVesselID\n")
        for index, node in enumerate(patient.Topology.OutletNodes):
            line = ",".join((str(node.Number),
                             (str(node.Position[0]) + " " + str(node.Position[1]) + " " + str(node.Position[2])),
                             str(0), str(index),
                             str(0),
                             str(2),))
            f.write(line + "\n")


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    patient_folder = arguments["<patient_folder>"]

    start_time = datetime.now()
    transcript.start(patient_folder + 'logfile.log')

    generatebloodflowfiles(patient_folder)
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    transcript.stop()
