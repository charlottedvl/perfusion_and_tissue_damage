import subprocess
import shutil
import os
import sys

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside container) for configuration files
perfusion_dir = 'pf_sim/'
TISSUE_ROOT = "/app/tissue_health/"
CONFIG_TISSUE = "config_tissue_damage.yaml"
CONFIG_TISSUE_PROPAGATION = "config_propagation.yaml"

class API(API):
    def event(self):

        # output paths
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")

        # read configuration
        if self.current_model.get('type') == 'TISSUE-HEALTH-PROPAGATION':
            tissue_health_config_file = str(self.result_dir.joinpath(CONFIG_TISSUE_PROPAGATION))
            simulation_file_name = 'perfusion.nii.gz'
        elif self.current_model.get('type') == 'TISSUE-HEALTH':
            tissue_health_config_file = str(self.result_dir.joinpath(CONFIG_TISSUE))
            simulation_file_name = 'perfusion.xdmf'
        else:
            print(f"No API has been evaluated for model: `{self.current_model}`.")
            sys.exit(1)

        print(tissue_health_config_file)
        solver_config = read_yaml(tissue_health_config_file)

        # baseline, stroke, treatment directories
        paths = []
        for event in self.events:
            path = self.patient_dir.joinpath(event.get('event'))
            path = path.joinpath(perfusion_dir)
            path = path.joinpath(simulation_file_name)
            paths.append(path)

        baseline_dir, stroke_dir, treatment_dir = paths

        # update configuration
        solver_config['input']['healthyfile'] = str(baseline_dir)
        solver_config['input']['strokefile'] = str(stroke_dir)
        solver_config['input']['treatmentfile'] = str(treatment_dir)
        solver_config['output']['res_fldr'] = str(res_folder)

        # Set arrival time from onset to treatment, the sum of `dur_oer`
        # and `dur_erg`, converted from minutes to hours.
        arrival_time = self.patient['dur_onset_groin'] / 60
        solver_config['input']['arrival_time'] = arrival_time

        # The recovery time is expressed as the interval in number of days
        # after which the tissue health is re-evaluated. The default value is
        # set to a single day if no value is provided in the virtual patient.
        #
        # TODO: we might want to support arrays of recovery times, i.e.
        # `recovery_time = [1, 5, 7]`, to allow for probing the difference in
        # tissue health across various measurement intervals after treatment
        # (see issue #11 in `insist-trials` repository).
        num_recovery_days = self.patient.get('tissue_health_follow_up_days', 1)
        solver_config['input']['recovery_time'] = num_recovery_days * 24
        solver_config['input']['mesh_file'] = str(self.result_dir.joinpath("bf_sim/clustered_mesh.xdmf"))

        # write the updated yaml to the patient directory
        write_yaml(tissue_health_config_file, solver_config)

        # the path where we are going to write the infarct volumes
        outcome_path = self.patient_dir.joinpath('tissue_health_outcome.yml')

        # initialise the simulation with the updated configuration files
        if self.current_model.get('type') == 'TISSUE-HEALTH-PROPAGATION':
            tissue_health_cmd = [
                "python3",
                "tissue_health_propagation.py",
                "--config_file",
                str(tissue_health_config_file),
                "--res_fldr",
                str(self.result_dir.joinpath(f"{perfusion_dir}")),
                "--res_yaml",
                str(outcome_path),

            ]
        elif self.current_model.get('type') == 'TISSUE-HEALTH':
            tissue_health_cmd = [
                "python3",
                "infarct_estimate_treatment.py",
                str(tissue_health_config_file),
                str(outcome_path),
            ]
        else:
            print(f"No API has been evaluated for model: `{self.current_model}`.")
            sys.exit(1)

        subprocess.run(tissue_health_cmd, check=True, cwd=TISSUE_ROOT)

    def example(self):
        self.event()

    def test(self):
        shutil.copy(os.path.join(TISSUE_ROOT, CONFIG_TISSUE), str(self.result_dir))
        self.example()
