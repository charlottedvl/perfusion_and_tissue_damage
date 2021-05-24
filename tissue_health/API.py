import subprocess

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside container) for configuration files
tissue_health_config_file = '/app/tissue_health/config_tissue_damage.yaml'
perfusion_dir = 'pf_sim'


class API(API):
    def event(self):

        # output paths
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")

        # read configuration for oxygen
        solver_config = read_yaml(tissue_health_config_file)

        # baseline, stroke, treatment directories
        paths = []
        for event in self.events:
            path = self.patient_dir.joinpath(event.get('event'))
            path = path.joinpath(perfusion_dir)
            path = path.joinpath('perfusion.xdmf')
            paths.append(path)

        baseline_dir, stroke_dir, treatment_dir = paths

        # update configuration
        solver_config['input']['healthyfile'] = str(baseline_dir)
        solver_config['input']['strokefile'] = str(stroke_dir)
        solver_config['input']['treatmentfile'] = str(treatment_dir)
        solver_config['output']['res_fldr'] = str(res_folder)

        # TODO: set arrival time and recovery times
        solver_config['input']['arrival_time'] = 3
        solver_config['input']['recovery_time'] = 120

        # write the updated yaml to the patient directory
        config_path = self.result_dir.joinpath(
                f'{perfusion_dir}/tissue_health_config.yaml')
        write_yaml(config_path, solver_config)

        # initialise the simulation with the updated configuration files
        tissue_health_cmd = [
                "python3", "infarct_estimate_treatment.py", str(config_path)]
        subprocess.run(tissue_health_cmd, check=True, cwd="/app/tissue_health")

    def example(self):
        pass

    def test(self):
        pass
