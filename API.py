from eventmodule import eventhandler
import os
import subprocess
import sys

# Default path (inside the container) pointing to the YAML configuration for
# the `basic_flow_solver` routine.
perfusion_config_file = '/app/perfusion/config_basic_flow_solver.yml'
blood_flow_dir = 'bf_sim'
perfusion_dir = 'pf_sim'
# filename used for boundary conditions in CSV format
bc_fn = 'boundary_condition_file.csv'


class API(eventhandler.EventHandler):
    def handle_event(self):
        perm_file = '/app/brain_meshes/b0000/permeability/K1_form.xdmf'

        if not os.path.exists(perm_file):
            permeability_cmd = ["python3", "permeability_initialiser.py"]

            print(
                f"Permeability file '{perm_file}' not present..."
                f"Evaluating: '{' '.join(permeability_cmd)}'",
                flush=True)

            subprocess.run(permeability_cmd, check=True, cwd="/app/perfusion")

        # output paths for perfusion simulation
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}/VP_result")
        os.makedirs(res_folder, exist_ok=True)

        # update configuration for perfusion
        solver_config = eventhandler.read_yaml(perfusion_config_file)

        # ensure boundary conditions are being read from input files
        solver_config['input']['read_inlet_boundary'] = True
        bc_file = self.result_dir.joinpath(f'{blood_flow_dir}/{bc_fn}')
        solver_config['input']['inlet_boundary_file'] = str(bc_file.resolve())

        # cannot proceed without boundary conditions
        msg = f"Boundary conditions `1d-blood-flow` not present: `{bc_file}`"
        assert os.path.isfile(bc_file), msg

        # update output settings
        config_path = self.result_dir.joinpath(
            f'{perfusion_dir}/perfusion_config.yml')
        eventhandler.write_yaml(solver_config, config_path)

        # form command to evaluate perfusion
        solve_cmd = [
            "python3", "basic_flow_solver.py", "--res_fldr", f"{res_folder}/",
            "--config_file", f"{str(config_path)}"
        ]

        print(f"Evaluating: '{' '.join(solve_cmd)}'", flush=True)
        subprocess.run(solve_cmd, check=True, cwd="/app/perfusion")

    def handle_example(self):
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
        self.handle_event()

    def handle_test(self):
        self.handle_example()


if __name__ == "__main__":
    api = API(sys.argv[1:]).evaluate()
