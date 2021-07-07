from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

from contextlib import suppress
import distutils.dir_util
import os
import pathlib
import shutil
import subprocess

# Default path (inside the container) pointing to the YAML configuration for
# the `basic_flow_solver` routine.
perfusion_config_file = '/app/perfusion_and_tissue_damage/perfusion/config_coupled_flow_solver.yaml'
blood_flow_dir = 'bf_sim'
perfusion_dir = 'pf_sim'
# filename used for boundary conditions in CSV format
bc_fn = 'boundary_condition_file.csv'
# filename used for perfusion output required for simplified infarct values
pf_outfile = 'perfusion.xdmf'


def dict_to_xml(dictionary):
    import xml.etree.ElementTree as ET
    import xml.dom.minidom

    root = ET.Element("virtualPatient")
    patient = ET.SubElement(root, "Patient")

    # Directly convert each (key, value) into an XML element, except for the
    # events. These are given as a list and require specific treament.
    for key, val in dictionary.items():

        if key in 'models' or key == 'events' or key == 'labels':
            continue

        # directly convert the key to XML element with text set to its value
        el = ET.SubElement(patient, key)
        el.text = str(val)

    xml_str = ET.tostring(root, encoding="unicode")
    dom = xml.dom.minidom.parseString(xml_str)
    return dom.toprettyxml()


def update_patient_anatomy(patient, input_file, output_file):
    """Update the patient anatomy with patient-specific properties.

    The ``input_file`` refers to the original ``1-D_Anatomy.txt`` that is used
    as a template for the output. The relevant lines are modified and the new
    contents of the file are written to the path at ``output_file``.
    """
    ica_tag = "int. carotid"
    m1_tag = "MCA"
    delim = '\t'

    # The file is read in memory completely, these are sufficiently small that
    # we do not need to do any line-by-line type processing.
    with open(input_file, 'r') as infile:
        contents = [line for line in infile]

    # the modified file starts with the same header
    modified_lines = [contents[0]]

    ica_len = str(patient['ica_length_mm'])
    ica_rad = str(patient['ica_rad_mm'])
    m1_rad = str(patient['m1_rad_mm'])

    # Ignore the header and only changes lines that match the `ica_tag` and
    # `m1_tag` which indicate a line corresponds to the ICA or M1 vessel. For
    # these vessels patient-specific information is available and has to be
    # included within the anatomy description.
    for line in contents[1:]:
        if ica_tag in line:
            s = line.split(delim)
            line = delim.join([s[0], s[1], ica_len, ica_rad, ica_rad, s[-1]])

        if m1_tag in line:
            s = line.split(delim)
            line = delim.join([s[0], s[1], s[2], m1_rad, m1_rad, s[-1]])

        modified_lines.append(line)

    with open(output_file, 'w') as outfile:
        for line in modified_lines:
            outfile.write(line)


class API(API):
    def event(self):
        # Convert the `patient.yml` configuration to XML format as the
        # 1d-blood-flow reads its configuration information from XML only.
        # The file is converted on every call to ensure most up to date config.
        xml_config_path = self.patient_dir.joinpath('config.xml')

        # update the XML configuration file
        xml_config = dict_to_xml(dict(self.patient))
        with open(xml_config_path, 'w') as outfile:
            outfile.write(xml_config)

        # the subdirectory for blood-flow specific result files
        sim_dir = self.result_dir.joinpath("bf_sim/")
        os.makedirs(sim_dir, exist_ok=True)

        # copy the patient's configuration file and clots to result directory
        copy_files_from_to = [
            (self.patient_dir.joinpath('config.xml'), self.result_dir),
            (self.patient_dir.joinpath('Clots.txt'), self.result_dir),
        ]
        for source, destination in copy_files_from_to:
            shutil.copy(source, destination)

        # When running a subsequent evaluation of the blood flow model, e.g.
        # for the stroke or treatment phases, the original, initialised files
        # of the patient can be copied to the current result directory. This
        # acts as a "warm" start to the model and avoids reinitialising various
        # files for the blood flow model, which were already in place from the
        # original, "baseline" simulation from previous evaluations.
        if self.previous_event is not None:
            # FIXME: from `python 3.8` replace with `shutil.copytree()`,
            # current version does not support `shutil.copytree()` when some of
            # the directories already exist.
            distutils.dir_util.copy_tree(
                str(self.previous_result_dir),
                str(self.result_dir)
            )

        # We only perform the initialisation if the `Run.txt` file is not
        # present. This will only be the case on the first evaluation of this
        # model, as subsequent evaluations will copy this file to the current
        # simulation directory `sim_dir` already. Thus, changing any of the
        # values in the following files will influence _all_ steps.
        if not (os.path.isfile(sim_dir.joinpath("Run.txt"))):

            # The root directory is found with respect to the current file's
            # position on the file system, to ensure it is relative either
            # inside or outside of the container.
            root = pathlib.Path('/app/bloodflow/DataFiles/DefaultPatient/')

            copy_default_files_from_to = [
                (root.joinpath('1-D_Anatomy.txt'), self.result_dir),
                (root.joinpath('bf_sim/labelled_vol_mesh.msh'), sim_dir),
                (root.joinpath('bf_sim/PialSurface.vtp'), sim_dir),
                (root.joinpath('bf_sim/Model_parameters.txt'), sim_dir),
            ]
            for source, destination in copy_default_files_from_to:
                # These files are only copied from the default patient
                # directory when they are *not* present on the expected
                # destination location. That should enable modification of such
                # parameter files by providing and alternative definition in
                # the expected location, which will then not be overwritten by
                # the defaults
                if not os.path.isfile(destination):
                    shutil.copy(source, destination)

            # Note: here we intervene with updating the patient anatomy. This
            # injects patient-specific anatomy information within the patient
            # anatomy definition. For instance, this updates the M1 and ICA
            # vessel radii.
            update_patient_anatomy(
                    self.patient,
                    self.result_dir.joinpath('1-D_Anatomy.txt'),
                    self.result_dir.joinpath('1-D_Anatomy_Patient.txt')
            )

            # This invokes the initialisation script to generate all require
            # patient files for the blood-flow simulation. Note, the
            # `1-D_Anatomy_Patient.txt` file is not updated anymore, as we
            # already generate this manually with patient-specific information
            # using the `update_patient_anatomy` function.
            subprocess.run([
                "python3",
                "bloodflow/generate_files.py",
                str(self.result_dir)
            ])

            # This cleans any large files that are require only during the
            # initialisation phase and not in any subsequent evaluations of
            # this model. This saves significant disk space.
            files_to_clean = [sim_dir.joinpath('Distancemat.npy')]
            for target in files_to_clean:
                with suppress(FileNotFoundError, IsADirectoryError) as _:
                    os.remove(target)

        perm_file = '/app/perfusion_and_tissue_damage/brain_meshes/b0000/permeability/K1_form.xdmf'

        if not os.path.exists(perm_file):
            permeability_cmd = ["python3", "permeability_initialiser.py"]

            print(
                f"Permeability file '{perm_file}' not present..."
                f"Evaluating: '{' '.join(permeability_cmd)}'",
                flush=True)

            subprocess.run(permeability_cmd, check=True, cwd="/app/perfusion_and_tissue_damage/perfusion")

        # output paths for perfusion simulation
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        # update configuration for perfusion
        solver_config = read_yaml(perfusion_config_file)

        # ensure boundary conditions are being read from input files
        solver_config['input']['read_inlet_boundary'] = True
        bc_file = self.result_dir.joinpath(f'{blood_flow_dir}/{bc_fn}')
        solver_config['input']['inlet_boundary_file'] = str(bc_file.resolve())

        # cannot proceed without boundary conditions
        #msg = f"Boundary conditions `1d-blood-flow` not present: `{bc_file}`"
        #assert os.path.isfile(bc_file), msg

        # update output settings
        config_path = self.result_dir.joinpath(
            f'{perfusion_dir}/perfusion_config.yaml')
        write_yaml(config_path, solver_config)

        # form command to evaluate perfusion
        solve_cmd = [
            "python3", "coupled_flow_solver.py", "--res_fldr", f"{res_folder}/",
            "--config_file", f"{str(config_path)}"
        ]

        print(f"Evaluating: '{' '.join(solve_cmd)}'", flush=True)
        subprocess.run(solve_cmd, check=True, cwd="/app/perfusion_and_tissue_damage/perfusion")

        # terminate baseline scenario
        if self.event_id == 0:
            return

        # extract settings
        if not self.model.get('evaluate_infarct_estimates', True):
            return

        labels = [event.get('event') for event in self.events]

        # baseline scenario result directories
        baseline = self.patient_dir.joinpath(labels[0])
        baseline = baseline.joinpath(perfusion_dir)
        baseline = baseline.joinpath(pf_outfile)

        # occluded scenario assumed to be the current result
        occluded = self.patient_dir.joinpath(labels[1])
        occluded = occluded.joinpath(perfusion_dir)
        occluded = occluded.joinpath(pf_outfile)

        for path in [baseline, occluded]:
            assert os.path.exists(path), f"File not found: '{path}'."

        # evaluate preliminary infarct volumes at multiple thresholds
        infarct_cmd = [
            "python3",
            "infarct_calculation_thresholds.py",
            "--config_file",
            f"{str(config_path)}",
            "--baseline",
            f"{str(baseline)}",
            "--occluded",
            f"{str(occluded)}",
            "--res_fldr",
            f"{res_folder}/",
            "--thresholds",
            f"{self.model.get('infarct_levels', 21)}",
        ]
        print(f"Evaluating: '{' '.join(infarct_cmd)}'", flush=True)
        subprocess.run(infarct_cmd, check=True, cwd="/app/perfusion_and_tissue_damage/perfusion")

    def example(self):
        # when running the example, we need to generate some dummy input
        # for the boundary conditions, for this, use the `BC_creator.py`
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        bc_cmd = [
            "python3",
            "BC_creator.py",
            "--occluded",
            "--res_fldr",
            f"{res_folder}/",
            "--config_file",
            f"{perfusion_config_file}",
            "--folder",
            f"{res_folder}/",
        ]

        print(f"Evaluating: '{' '.join(bc_cmd)}'", flush=True)
        subprocess.run(bc_cmd, check=True, cwd="/app/perfusion_and_tissue_damage/perfusion")

        # rename the boundary conditions file to match the trial scenario
        src = res_folder.joinpath('BCs.csv')
        dst = self.result_dir.joinpath(f"bf_sim/{bc_fn}")

        os.makedirs(dst.parent, exist_ok=True)
        os.rename(src, dst)

        # run event with example boundary conditions
        self.handle_event()

    def test(self):
        self.handle_example()
