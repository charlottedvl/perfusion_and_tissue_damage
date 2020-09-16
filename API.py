from eventmodule import eventhandler
import os
import subprocess
import sys


class API(eventhandler.EventHandler):
    def handle_event(self):
        perm_file = '/app/brain_meshes/b0000/permeability/K1_form.xdmf'
        if not os.path.exists(perm_file):
            subprocess.run(["python3", "permeability_initialiser.py"],
                           check=True,
                           cwd="./perfusion")

        res_fldr = self.patient_dir.joinpath("VP_result")

        subprocess.run(
            ["python3", "BC_creator.py", "--res_fldr", f"{res_fldr}/"],
            check=True,
            cwd="./perfusion")

        subprocess.run(
            ["python3", "basic_flow_solver.py", "--res_fldr", f"{res_fldr}/"],
            check=True,
            cwd="./perfusion")

    def handle_example(self):
        self.handle_event()

    def handle_test(self):
        self.handle_example()


if __name__ == "__main__":
    api = API(sys.argv[1:]).evaluate()
