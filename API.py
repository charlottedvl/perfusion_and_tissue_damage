from perfusion.API import API as perfusion_API
from oxygen.API import API as oxygen_API
import sys


if __name__ == "__main__":
    api = perfusion_API(sys.argv[1:])
    analysis = api.model['type']

    if analysis == 'PERFUSION':
        sys.exit(perfusion_API(sys.argv[1:]).evaluate())

    if analysis == 'OXYGEN':
        sys.exit(oxygen_API(sys.argv[1:]).evaluate())

    sys.exit("No model was evaluated.")
