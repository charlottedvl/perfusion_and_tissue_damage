import subprocess

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside container) for configuration files
oxygen_config_file = '/app/oxygen/config_oxygen_solver.yaml'
perfusion_config_file = '/app/perfusion/config_basic_flow_solver.yaml'
perfusion_dir = 'pf_sim'


class API(API):
    def event(self):

        # output paths
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")

        # read properties from previous perfusion step
        perfusion_config = read_yaml(
            f'{res_folder}/perfusion_config.yaml')
        bc_file = perfusion_config.get('input', {}).get('inlet_boundary_file')
        assert bc_file is not None, "no path for inlet boundary conditions"

        # read configuration for oxygen
        solver_config = read_yaml(oxygen_config_file)

        # update configuration
        solver_config['input']['para_path'] = f'{res_folder.resolve()}/'
        solver_config['input']['read_inlet_boundary'] = True
        solver_config['input']['pialBC_file'] = bc_file

        # write configuration to disk
        config_path = self.result_dir.joinpath(
            f'{perfusion_dir}/oxygen_config.yaml')
        write_yaml(config_path, solver_config)

        # setup command for oxygen model with arguments
        oxygen_cmd = [
            "python3", "oxygen_main.py", "--config_file",
            f"{str(config_path)}", "--rslt", f"{res_folder}/"
        ]

        # invoke oxygen model
        print(f"Evaluting: '{' '.join(oxygen_cmd)}'", flush=True)
        subprocess.run(oxygen_cmd, check=True, cwd="/app/oxygen")

    def example(self):
        self.event()

    def test(self):
        self.example()
