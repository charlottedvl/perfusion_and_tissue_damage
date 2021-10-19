import os
import pathlib
import subprocess

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside the container) pointing to the YAML configuration for
# the `basic_flow_solver` routine.
permeability_config_file = 'config_permeability_initialiser.yaml'
perfusion_config_file = 'config_basic_flow_solver.yaml'
blood_flow_dir = 'bf_sim'
perfusion_dir = 'pf_sim'
# filename used for boundary conditions in CSV format
bc_fn = 'boundary_condition_file.csv'
# filename used for perfusion output required for simplified infarct values
pf_outfile = 'perfusion.xdmf'
# directory for where to generate and store patient brain meshes
brain_mesh_dir = 'brain_meshes'


def initialise_brain_mesh(patient_dir, out_dir, cwd):
    brain_mesh = patient_dir.joinpath(brain_mesh_dir)
    clustered_mesh = out_dir.joinpath(f'{blood_flow_dir}/clustered_mesh.msh')

    if brain_mesh.exists():
        return 0

    brain_mesh_cmd = [
            "python3",
            "convert_msh2hdf5.py",
            str(clustered_mesh),
            str(brain_mesh.joinpath('clustered')),
    ]

    print(f"Evaluating: '{' '.join(brain_mesh_cmd)}'", flush=True)
    subprocess.run(brain_mesh_cmd, check=True, cwd=str(cwd))

    error_msg = f"""The brain mesh or clustered files are missing on the
    file system although the initialisation has been evaluated. This is not
    supported. Please investigate: {brain_mesh} and {clustered_mesh}
    paths."""
    assert brain_mesh.exists() and clustered_mesh.exists(), error_msg


def initialise_permeability(patient_dir, result_dir, cwd):
    brain_mesh = patient_dir.joinpath(brain_mesh_dir)
    permeability_dir = brain_mesh.joinpath('permeability')

    if permeability_dir.exists():
        return 0

    clustered_xdmf = brain_mesh.joinpath('clustered.xdmf')
    config_path = result_dir.joinpath(f'{perfusion_dir}/perm_config.yaml')

    config = read_yaml(f'{cwd}/{permeability_config_file}')
    config['input']['mesh_file'] = str(clustered_xdmf)
    config['output']['res_fldr'] = f'{permeability_dir}/'
    write_yaml(config_path, config)

    permeability_cmd = [
        "python3",
        "permeability_initialiser.py",
        "--config_file",
        str(config_path)
    ]

    print(f"Evaluating: '{' '.join(permeability_cmd)}'", flush=True)
    subprocess.run(permeability_cmd, check=True, cwd=str(cwd))

    error_msg = f"""The permeability files do not exist, which should have
    been generated at '{permeability_dir}' using
    'permeability_initialiser.py' and its configuration at
    '{config_path}'."""
    assert permeability_dir.exists(), error_msg


class API(API):
    def __init__(self, patient, model_id, coupled=False, **kwargs):
        super().__init__(patient, model_id)
        self.coupled = coupled

    def event(self):
        if self.coupled:
            cwd = pathlib.Path('/app/perfusion_and_tissue_damage/perfusion')
            solver = "coupled"
        else:
            cwd = pathlib.Path('/app/perfusion')
            solver = "basic"

        # ensure the clustered brain mesh is present
        initialise_brain_mesh(self.patient_dir, self.result_dir, cwd)
        initialise_permeability(self.patient_dir, self.result_dir, cwd)

        # output paths for perfusion simulation
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        # update configuration for perfusion
        perfusion_config_file = "config_{}_flow_solver.yaml".format(solver)
        config = read_yaml(cwd.joinpath(perfusion_config_file))

        # ensure boundary conditions are being read from input files
        config['input']['read_inlet_boundary'] = True
        bc_file = self.result_dir.joinpath(f'{blood_flow_dir}/{bc_fn}')
        config['input']['inlet_boundary_file'] = str(bc_file.resolve())
        mesh_file = self.patient_dir.joinpath('brain_meshes/clustered.xdmf')
        config['input']['mesh_file'] = str(mesh_file)
        perm_file = self.patient_dir.joinpath('brain_meshes/permeability')
        config['input']['permeability_folder'] = f"{perm_file}/"
        model_type = self.current_model.get('model_type', 'a')
        config['simulation']['model_type'] = model_type
        coupled_model = self.current_model.get('coupled_model', True)
        config['simulation']['coupled_model'] = coupled_model
        config['output']['res_fldr'] = str(res_folder)

        # FIXME: this is only for testing purposes
        fe_degr = int(self.current_model.get('finite_element_degree', 2))
        config['simulation']['fe_degr'] = fe_degr
        print(f"Running with finite element degree: {fe_degr}", flush=True)

        # cannot proceed without boundary conditions
        if not self.coupled:
            msg = f"""Boundary conditions `1d-blood-flow` not present:
                `{bc_file}`"""
            assert os.path.isfile(bc_file), msg

        # update output settings
        config_path = self.result_dir.joinpath(
            f'{perfusion_dir}/perfusion_config.yaml')
        write_yaml(config_path, config)

        # form command to evaluate perfusion
        solve_cmd = [
            "python3",
            "{}_flow_solver.py".format(solver),
            "--res_fldr", f"{res_folder}/",
            "--config_file", f"{str(config_path)}"
        ]

        if self.coupled:
            if self.event_id == 0:
                solve_cmd.append('--healthy_scenario')
            if self.patient_dir.joinpath('clot_present').exists():
                solve_cmd.append('--clot_present')

        print(f"Evaluating: '{' '.join(solve_cmd)}'", flush=True)
        subprocess.run(solve_cmd, check=True, cwd=str(cwd))

        # terminate baseline scenario
        if self.event_id == 0:
            return

        # extract settings
        if not self.current_model.get('evaluate_infarct_estimates', False):
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
            f"{self.current_model.get('infarct_levels', 21)}",
            "--mesh_file",
            str(self.patient_dir.joinpath('brain_meshes/clustered.xdmf'))
        ]
        print(f"Evaluating: '{' '.join(infarct_cmd)}'", flush=True)
        subprocess.run(infarct_cmd, check=True, cwd=cwd)

    def example(self):
        # when running the example, we need to generate some dummy input
        # for the boundary conditions, for this, use the `BC_creator.py`
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        # The current event simulation does not require the brain meshes and
        # therefore the default container does not contain the _extracted_
        # meshes by default. So, when running in CI/CD we extract the
        # compressed archive manually before running the event.
        import tarfile
        tar = tarfile.open("/app/brain_meshes.tar.xz", "r:xz")
        tar.extractall("/patient/tmp")
        tar.close()

        # move the contents to the expected brain mesh location
        import shutil
        shutil.move("/patient/tmp/brain_meshes/b0000", "/patient/brain_meshes")

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
            "--mesh_file",
            "/patient/brain_meshes/clustered.xdmf",
        ]

        print(f"Evaluating: '{' '.join(bc_cmd)}'", flush=True)
        subprocess.run(bc_cmd, check=True, cwd="/app/perfusion")

        # rename the boundary conditions file to match the trial scenario
        src = res_folder.joinpath('BCs.csv')
        dst = self.result_dir.joinpath(f"bf_sim/{bc_fn}")

        os.makedirs(dst.parent, exist_ok=True)
        os.rename(src, dst)

        # run event with example boundary conditions
        self.event()

    def test(self):
        self.example()
