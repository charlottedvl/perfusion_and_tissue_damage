import os
import subprocess

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside the container) pointing to the YAML configuration for
# the `basic_flow_solver` routine.
permeability_config_file = '/app/perfusion/config_permeability_initialiser.yaml'
perfusion_config_file = '/app/perfusion/config_basic_flow_solver.yaml'
blood_flow_dir = 'bf_sim'
perfusion_dir = 'pf_sim'
# filename used for boundary conditions in CSV format
bc_fn = 'boundary_condition_file.csv'
# filename used for perfusion output required for simplified infarct values
pf_outfile = 'perfusion.xdmf'
# directory for where to generate and store patient brain meshes
brain_mesh_dir = 'brain_meshes'


class API(API):
    def event(self):

        # patient's brain meshes and permeability information
        brain_meshes = self.patient_dir.joinpath(brain_mesh_dir)
        permeability_dir = brain_meshes.joinpath('permeability')

        if not brain_meshes.exists() or not permeability_dir.exists():
            error_msg = f"""Brain meshes and permeability files are not present
            although previous events have been evaluated. This is not supported
            and requires investigation why {brain_meshes} or {permeability_dir}
            are not present anymore on the system."""
            assert self.event_id == 0, error_msg

            # generate clustering files from blood flow output
            clustered_mesh = self.result_dir.joinpath(f'{blood_flow_dir}/clustered_mesh.msh')
            clustering_result_dir = brain_meshes.joinpath('clustered')

            brain_mesh_cmd = [
                    "python3",
                    "convert_msh2hdf5.py",
                    str(clustered_mesh),
                    str(clustering_result_dir)
            ]
            print(f"Evaluating: '{' '.join(brain_mesh_cmd)}'", flush=True)
            subprocess.run(brain_mesh_cmd, check=True, cwd="/app/perfusion")

            # generate permeability meshes after clustering
            clustered_mesh = brain_meshes.joinpath('clustered.xdmf')

            perm_config = read_yaml(permeability_config_file)
            perm_config['input']['mesh_file'] = str(clustered_mesh)
            perm_config['output']['res_fldr'] = f'{permeability_dir}/'

            config_path = self.result_dir.joinpath(f'{perfusion_dir}/perm_config.yaml')
            write_yaml(config_path, perm_config)

            permeability_cmd = [
                "python3",
                "permeability_initialiser.py",
                "--config_file",
                str(config_path)
            ]
            print(f"Evaluating: '{' '.join(permeability_cmd)}'", flush=True)
            subprocess.run(permeability_cmd, check=True, cwd="/app/perfusion")

        assert brain_meshes.exists(), f"Brain meshes not at: '{brain_meshes}'."

        # output paths for perfusion simulation
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        # update configuration for perfusion
        solver_config = read_yaml(perfusion_config_file)

        # ensure boundary conditions are being read from input files
        solver_config['input']['read_inlet_boundary'] = True
        bc_file = self.result_dir.joinpath(f'{blood_flow_dir}/{bc_fn}')
        solver_config['input']['inlet_boundary_file'] = str(bc_file.resolve())
        solver_config['input']['mesh_file'] = str(self.patient_dir.joinpath('brain_meshes/clustered.xdmf'))
        solver_config['input']['permeability_folder'] = f"{self.patient_dir.joinpath('brain_meshes/permeability')}/"

        # cannot proceed without boundary conditions
        msg = f"Boundary conditions `1d-blood-flow` not present: `{bc_file}`"
        assert os.path.isfile(bc_file), msg

        # update output settings
        config_path = self.result_dir.joinpath(
            f'{perfusion_dir}/perfusion_config.yaml')
        write_yaml(config_path, solver_config)

        # form command to evaluate perfusion
        solve_cmd = [
            "python3", "basic_flow_solver.py", "--res_fldr", f"{res_folder}/",
            "--config_file", f"{str(config_path)}"
        ]

        print(f"Evaluating: '{' '.join(solve_cmd)}'", flush=True)
        subprocess.run(solve_cmd, check=True, cwd="/app/perfusion")

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
        subprocess.run(infarct_cmd, check=True, cwd="/app/perfusion")

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
